# Renderer Improvement Roadmap

> **Guiding principle:** tackle items in order of _(impact → risk → effort)_ so the engine stays usable and you avoid re-work.

---

## 0 · Instrument & Baseline  
* Add a **micro-profiler** (GPU events + host timers).  
* Insert **frame-exit validation** (`cudaGetLastError`, etc.).

_Time: ≈½ day – unlocks reliable speed-up numbers._

---

## 1 · Eliminate Per-Frame Graph Overhead 🚀  
1. **Reuse one `CUgraphExec`:** build the graph once; patch args each frame with `cuGraphExecKernelNodeSetParams`.  
2. **Stream-capture the first frame:**  
   * `cuStreamBeginCapture_v2 … cuStreamEndCapture` records your current sequence.  
   * Dependencies are inferred automatically.

_Time: 1-2 days – usually 5–30 % frame-time win._

---

## 2 · Unlock Intra-Frame Concurrency  
* Capture independent kernels into **separate streams** inside the same graph.  
* If dependencies vary, switch to **conditional nodes** (CUDA 12.8 +).

_Time: ≈1 day – hides latency for short kernels._

---

## 3 · Event-Driven Render Loop  
1. Move **`Camera` into ECS**.  
2. Keep a **dirty-flag**; if neither camera nor any `Renderable` changed, skip ray generation & march (just blit last texture).

_Time: 1 day – huge savings during scene editing._

---

## 4 · Sampling Efficiency Experiments  
1. Benchmark **megakernel SPP vs N-launch SPP**.  
2. Implement a **statistics-based adaptive sampler** (per-pixel variance buffer → adaptive SPP per tile).

_Time: 2-3 days – can halve sample counts on noisy shots._

---

## 5 · Foundational Ray Tracing Features  
1. **First ray-tracing kernel** (BVH traversal, basic shading).  
2. **glTF importer** to feed triangle meshes.

_Time: ~1 week – enables real-world scenes._

---

## 6 · Path Tracing Core  
* Extend the ray-tracer to full PT (recursive/wavefront, MIS, Russian roulette).

_Time: 2–3 weeks._

---

## 7 · Basic AI Post-Processing  
1. **Denoiser** (OIDN/OptiX or custom tiny CNN).  
2. **Neural SDF** prototypes to replace analytic SDFs.

_Can run in parallel with Stage 6._

---

## 8 · Physics & Advanced AI  
* Add **basic physics** (Euler, broad-phase colliders) inside ECS.  
* Explore **PINN-based simulations** once the renderer is stable for visualisation.

---

### Why This Order?

| Stage | Immediate Benefit | Blocks Later Work? | Risk |
|-------|------------------|--------------------|------|
| **1** | Removes biggest CPU bottleneck | Yes (benchmarks) | Very low |
| **2** | Latency hiding | Needed for SPP tests | Low |
| **3** | Cuts wasted frames | Cleans architecture | Low |
| **4** | Finds best sampling strategy | Influences PT | Medium |
| **5** | Mesh content | Required for PT | Medium |
| **6** | Photorealism | Core engine | High |
| **7** | Visual quality | Independent | Low-Med |
| **8** | Gameplay / research | Nice-to-have | High |

Follow this path to **measure → accelerate → organise → extend**—each layer resting on a profiled, stable foundation.
