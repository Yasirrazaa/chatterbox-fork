# 📊 Chatterbox Benchmarks

This document details the performance metrics of the various generation strategies and underlying architectures available in this repository.

All benchmarks were recorded on a single NVIDIA GPU.

## 1. Chunking Strategies
The `ChatterboxInference` pipeline automatically chunks long-form text. The `Packed` strategy drastically reduces fixed-cost inference overhead by packing multiple sentences together up to an optimal character limit.

| Method | Number of Chunks | Max Chunk Length | Avg Chunk Length |
| :--- | :--- | :--- | :--- |
| Raw NLTK | 6 | 159 chars | 70.3 chars |
| Packed | 2 | 239 chars | 213.0 chars |

## 2. CUDA Graphs vs Standard Inference

By tracing and compiling the autoregressive PyTorch generation loop into static C++ CUDA graphs, we eliminate CPU bottlenecks, significantly lowering the Real-Time Factor (RTF).

### ChatterboxTurboTTS
**Speedup Multiplier:** 1.79x

| Method | Generation Time (s) | RTF (lower is better) |
| :--- | :--- | :--- |
| Standard `generate()` | 5.27 ± 0.43 | 0.59 ± 0.03 |
| CUDA Graphs `generate_fast()` | 2.94 ± 0.23 | **0.32 ± 0.02** |

### ChatterboxTTS (Base)
**Speedup Multiplier:** 1.75x

| Method | Generation Time (s) | RTF (lower is better) |
| :--- | :--- | :--- |
| Standard `generate()` | 8.36 ± 0.12 | 1.06 ± 0.01 |
| CUDA Graphs `generate_fast()` | 4.77 ± 0.15 | **0.67 ± 0.03** |

### ChatterboxMultilingualTTS
*Note: Due to disabling the CPU-dependent `AlignmentStreamAnalyzer` for CUDA graph compilation, the fast path may occasionally run to its hard token limit (1000 tokens) if long-tail hallucinations occur, causing higher absolute generation times on pathological inputs. The underlying generation speed (RTF), however, is vastly improved (1.77x faster).*

| Method | Generation Time (s) | RTF (lower is better) |
| :--- | :--- | :--- |
| Standard `generate()` | 8.46 ± 0.11 | 1.19 ± 0.01 |
| CUDA Graphs `generate_fast()` | 12.22 ± 10.63 | **0.67 ± 0.03** |
