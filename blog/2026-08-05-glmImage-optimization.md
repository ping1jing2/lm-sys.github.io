---
title: "Full-Stack Performance Optimization of AR+DiT in SGL-Diffusion"
author: "Ascend Team"
date: "August 05, 2026"
previewImg: /images/blog/2026-08-05-glmImage-optimization/01-cover.png
type: blog
---

## TL;DR

- Replaces the HF backend with SRT to accelerate AR modeling and resolve parallelism conflicts, with dedicated TP for AR and SP for DiT
- Boosts hardware utilization via dynamic batching and enables early return for completed images
- Implements one-denoiser-per-device parallel DiT execution and overlaps AR & DiT workflows via cached AR results

<div align="center">
  <img src="./asserts/02-peformance_result.jpg" alt="performance result" />
  <br>
  <em>Figure 1: Performance comparison.</em>
</div>

## 1. Background

Hybrid autoregressive–diffusion (AR+DiT) generation is a unified framework that combines AR modeling for global context with DiT methods for local detail refinement. It leverages AR transformers to capture long-range dependencies while DiT models iteratively refine outputs, ensuring improved quality and efficiency. GLM-Image exemplifies this trend: a 9B vision-language model first autoregressively generates semantic prior tokens from a text prompt, and a 7B DiT then denoises those tokens into a high-resolution image over 30–50 steps. This "plan-then-paint" design delivers SOTA results on knowledge-intensive and text-heavy visual tasks—such as posters, infographics, and precise typography—where end-to-end diffusion models often struggle.

Yet serving such hybrid pipelines efficiently in SGLang exposes a fundamental tension. In the native deployment, the AR encoder, DiT denoiser, and VAE decoder are chained inside a single monolithic worker process, which creates three critical pain points:

1. **Architectural coupling.** AR and DiT share the same process, weight-loading lifecycle, and scheduling domain. Scaling up one stage necessarily drags the other along; there is no way to provision resources independently. AR is fundamentally an LLM-decode workload—throughput scales with batch size and tensor parallelism (TP). DiT denoising, by contrast, is a large-tensor, per-image computation that favors spatial parallelism (SP) across cards and is most efficient at batch=1. A homogeneous deployment must pick a single strategy, guaranteeing that one stage is always sub-optimal.
2. **Low resource utilization under concurrency.** Without dynamic batching, concurrent requests are processed serially. End-to-end latency grows almost linearly with request concurrency, leaving a large fraction of the available compute idle.
3. **Mismatched resource allocation.** DiT achieves its best per-request latency at batch=1 per device, but bundling all devices into a single monolithic pipeline forces DiT to run in a multi-card spatial-parallel configuration even when throughput is the priority, resulting in underutilized hardware capacity.

To resolve these issues, we contributed three progressively staged PRs that evolve the system from a monolith to a fully decoupled, heterogeneous distributed architecture:

<div align="center">
  <img src="./asserts/03-whole-pipeline.png" alt="the whole pipeline" />
  <br>
  <em>Figure 2: The whole pipeline for our optimization.</em>
</div>

## 2. SRT-ifying the AR Backend (PR #25381)

PR #25381 decouples the AR stage from the diffusion worker process into a standalone SRT service, so AR and DiT load weights separately, have decoupled scheduling lifecycles, and can scale independently. Meanwhile, the AR server can now configure TP on its own, no longer constrained by DiT's SP strategy.

<div align="center">
  <img src="./asserts/04-glm_image_ar.png" alt="glm image AR" />
  <br>
  <em>Figure 3: The comparison between Original and Target.</em>
</div>

The AR vision-language encoder is spun up as a standard SGLang SRT service, invoked remotely by the Diffusion pipeline via a new `--srt-encoder-url` option. The AR server reuses SGLang's existing multimodal `sglang serve` capability (`srt/models/glm_image_vl.py` + `srt/multimodal/processors/glm_image.py`); the Diffusion side only adds an HTTP `/generate` branch to the `GlmImageAR` stage, and `VisionLanguageEncoderLoader` simply performs a `/health` check and returns the URL when `srt-encoder-url` is set, instead of calling `from_pretrained` to load weights. This "standalone server + HTTP" approach turns the giant task of "rewriting a VLM into SRT" into "reuse existing infrastructure + one HTTP call," dramatically reducing coupling and minimizing invasive changes to the diffusion pipeline.

**Performance gains** (please refer to [PR #25381 description](https://github.com/sgl-project/sglang/pull/25381) for reproducing):

| Configuration                                  |         E2E latency |            AR stage | Denoising | Decoding |
| ---------------------------------------------- | ------------------: | ------------------: | --------: | -------: |
| baseline 1 NPU                                 |             154.6 s |             122.8 s |    31.6 s |  0.046 s |
| baseline 2 NPU (SP=2)                          |     144.9 s (−6.2%) |     127.9 s (+4.1%) |    16.9 s |  0.027 s |
| **new 1 NPU (SRT, TP=1)**                      | **78.3 s (−49.4%)** | **46.6 s (−62.1%)** |    31.6 s |  0.035 s |
| **new 4 NPU (SRT, TP=4 for AR, SP=4 for DiT)** | **35.2 s (−77.2%)** | **26.1 s (−78.8%)** |     9.0 s |  0.009 s |

The AR stage sees the most dramatic speedup: single-card 122.8 s → 26.1 s, a 78.8% reduction on 4 cards. Even at TP=1, SRT's CUDA Graph, continuous batching, and memory reuse deliver a −62.1% gain over the naive transformers `generate`. Notably, the baseline 2-NPU setup uses SP to cut denoising by 46.7%, but AR actually slows by 4.1% — under the old path AR gets zero benefit from SP and even regresses due to communication overhead; only the SRT path lets AR truly leverage multi-card TP.

## 3. Dynamic Batching Adaptation and Early Return Support (PR #30683)

After the separation, AR and DiT still execute one request at a time, so latency grows linearly under high concurrency (issue #30634). PR #30683 packs concurrent requests into single forward passes, eliminating the idle compute caused by serial execution.

1. **Dynamic batching adaptation**: SGL-Diffusion already includes a generic dynamic batching infrastructure (introduced in [PR #18764](https://github.com/sgl-project/sglang/pull/18764)); our work extends this capability to GLM-Image by implementing the `supports_dynamic_batching` and `supports_native_grouped_requests` interfaces and associated pipeline logic. After evaluation, we apply batching only to the AR stage, as DiT per-step latency scales proportionally with batch size and yields no net throughput benefit.
2. **Support early return**: we add the `num_grouped_prefix_stages` variable and related functions to support early return when each output image is ready, instead of waiting for the entire batch to finish.

**Performance gains** (please refer to [PR #30683 description](https://github.com/sgl-project/sglang/pull/30683) for reproducing):

| Metric                              | BS1    | BS4                 | BS8                     | BS16         |
| ----------------------------------- | ------ | ------------------- | ----------------------- | ------------ |
| **Throughput (img/s)**              | 0.0291 | 0.0519              | 0.0596                  | 0.0648       |
| Per‑request processing latency (s)¹ | 33.6   | 36 → 49 → 61 → 77.2 | 38.5 → 51.7 → … → 127.5 | 42 → … → 247 |
| AR stage per request (s)            | 20.17  | 5.65                | 3.20                    | 1.85         |
| Denoising per request (s)           | 12.58  | 12.73               | 12.69                   | 12.69        |
| Denoising per step (s)              | 0.0419 | 0.0424              | 0.0423                  | 0.0423       |
| Decoding per request (s)            | 0.33   | 0.39                | 0.38                    | 0.39         |
| Peak NPU memory (MB)                | 28 163 | 28 046              | 28 052                  | 28 062       |

**Notes:**  
¹ Processing latency is measured from batch dispatch to individual request completion. For BS4/BS8/BS16, the values represent a latency range across the batch: the first number corresponds to the fastest-finishing request, and the last to the slowest. Additional queueing wait time (≤14 ms in this test) is negligible.

## 4. Disaggregation and AR-to-DiT Fan-Out Architecture (PR #31320)

Fully decouple the two stages so AR and DiT each adopt the parallelism and deployment strategy that suits them best. The AR encoder favors large batch + TP (throughput-oriented); DiT denoising is optimal at batch=1 on a single NPU for both latency and throughput. Then #31320 introduces a heterogeneous topology: one batched AR server + a pool of independent batch=1 denoisers.

<div align="center">
  <img src="./asserts/05-fanout.png" alt="Disaggregated" />
  <br>
  <em>Figure 4: Final Deployment Architecture Diagram.</em>
</div>


SGL-Diffusion provides a generic disaggregation framework; PR #31320 adapts this framework to GLM-Image’s two-stage topology, enabling parallel DiT execution and pipeline overlap between AR generation and denoising. A key design choice is that only request metadata and CPU-side prior token IDs are transferred over ZMQ — no large tensors, latents, embeddings, or GPU buffers are sent across nodes — keeping communication overhead extremely low.

**Performance gains** (please refer to [PR #31320 description](https://github.com/sgl-project/sglang/pull/31320) for reproducing):

| Configuration                  |  NPU | Steady-state throughput | Median latency | 640-image time |
| ------------------------------ | ---: | ----------------------: | -------------: | -------------: |
| TP=2 AR + 14 batch=1 denoisers |   16 |        **0.69 image/s** |          ~74 s |    15 min 23 s |

## 5. Acknowledgments

- Huawei Ascend Team

  We thank the Huawei Ascend NPU team for its continued contributions to GLM-Image optimization. In particular, we recognize Maksim Emelin (@[Makcum888e](https://github.com/Makcum888e)), Artem Savkin (@[OrangeRedeng](https://github.com/OrangeRedeng)), Yuefeng Wu (@[ChefWu551](https://github.com/ChefWu551)), and Qianqian Zheng (@[AuFlow](https://github.com/AuFlow)), Liang Zhen (@[ping1jing2](https://github.com/ping1jing2)).

- SGLang Community

  We are grateful to the broader SGLang community, including code review from Xiaoyu Zhang (@[BBuf](https://github.com/BBuf)), and initial discussion (issue #20032) and implementation (PR #18809) from Yuhao Yang (@[yhyang201](https://github.com/yhyang201)) and other contributors.

Finally, we thank the SGLang maintainers and reviewers for their careful guidance, the Zhipu AI team for open-sourcing the GLM-Image model and weights, and everyone who has contributed to SGL-Diffusion.
