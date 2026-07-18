# 🚀 Chatterbox Studio (High-Performance Serverless & UI Edition)

Welcome to the **most optimized, feature-rich fork** of Resemble AI's state-of-the-art Chatterbox Text-to-Speech models.

This repository transforms the base Chatterbox research models into a **production-ready serverless engine**, integrating massive speedups, zero-hallucination pipelines, and beautiful web interfaces into a single, unified monorepo.

Whether you're running it locally through our custom Gradio Studio or deploying it as an ultra-fast API to a Cloud GPU (RunPod), this repository represents the absolute cutting-edge of open-source TTS engineering.

---

## 🌟 Why Use This Fork? (Key Features)

### 1. ⚡ 4x Faster Inference via Bucketed CUDA Graphs
Autoregressive TTS models are notoriously CPU-bound for non-batched inference (e.g., real-time voice agents). We solved this by integrating **Bucketed CUDA Graphs**.
* **Result:** **2x - 4x speedup** on batch-size 1 requests by pre-compiling the entire generation loop into static CUDA graphs and completely bypassing PyTorch CPU overhead.

### 2. 🛡️ Zero-Hallucination Pipeline (Whisper Validation)
TTS models occasionally hallucinate on tricky names or weird punctuation. We built an advanced pipeline to natively solve this.
* **Result:** Production-safe audio. By setting `num_candidates=3`, the engine generates multiple audio variations simultaneously, transcribes them in parallel using an **in-memory cached Faster-Whisper** model, and automatically selects the chunk with the lowest Word Error Rate (WER).

### 3. 🧠 Intelligent Long-Text Processing
* **Pre-processing**: Automatically cleans artifacts, removes filler words, and normalizes punctuation.
* **Sentence Chunking & Stitching**: Uses NLTK to automatically split massive paragraphs into digestible chunks, packs them up to an optimal character limit, and flawlessly stitches the generated audio back together.

### 4. 🎨 Chatterbox Studio (Unified Gradio UI)
A beautiful, highly-customized unified Gradio application that exposes all advanced parameters (Turbo tags, Voice Cloning, Voice Conversion, and Long-Form batching) across 5 interactive tabs.

---

## 📂 Project Architecture & File Structure

The project is structured to strictly separate the raw models from the high-level inference wrappers and user interfaces.

```text
chatterbox/
├── src/chatterbox/                 # Raw PyTorch Models (TTS, Turbo, Multilingual, VC)
├── src/chatterbox_serverless/      # High-Level Wrappers (Chunking, Validation, Inference)
├── examples/
│   ├── app.py                      # The unified Gradio Studio web interface
│   └── quickstart.py               # Consolidated Python API examples for all models
├── scripts/
│   ├── benchmark.py                # Comprehensive performance benchmarking suite
│   ├── local_test.py               # CLI tool to locally simulate a RunPod API request
│   └── data/                       # Difficult & Long text datasets for benchmarks
├── rp_handler.py                   # RunPod Serverless execution entrypoint
└── Dockerfile.serverless           # Production-ready Dockerfile for cloud deployment
```

**Architecture Flow:**
`app.py` / `rp_handler.py` ➡️ `ChatterboxInference` (handles chunking & validation) ➡️ `ChatterboxTurboTTS` (handles raw tensor generation via CUDA graphs).

---

## 🛠️ Installation

```shell
# 1. Clone the repository
git clone https://github.com/Yasirrazaa/chatterbox.git
cd chatterbox

# 2. Install dependencies (Using uv is highly recommended for speed)
pip install uv
uv venv
source .venv/bin/activate

# 3. Install the package
uv pip install -e .
```

---

## 🎮 Usage

### 1. The Gradio Studio (No Code)
The easiest way to test everything is through our unified web interface:
```shell
uv run examples/app.py
```
*This will launch a local web server with tabs for Turbo TTS, Multilingual TTS, Voice Conversion, and Long-Form Batching.*

### 2. Python API (Quickstart)
We've consolidated all developer examples into a single file. You can run it directly:
```shell
uv run examples/quickstart.py
```

Or import the high-performance pipeline into your own code:
```python
from chatterbox_serverless.inference import ChatterboxInference

# Load the model with optimizations (Downloads automatically from HuggingFace)
pipeline = ChatterboxInference.from_pretrained(model_type="turbo", device="cuda")

# Generate with text pre-processing, chunking, CUDA graphs, and Whisper validation
wav = pipeline.generate_fast(
    "Wow! That's incredibly fast. And it never hallucinates anymore!",
    language_id="en",
    normalize_text=True,
    sentence_split=True,
    inter_sentence_silence_ms=100,
    num_candidates=3,          # Generates 3 variations, picks the best WER!
)
```

---

## ☁️ Cloud Deployment (RunPod Serverless)

This repository comes natively equipped with a highly optimized RunPod Serverless worker (`rp_handler.py`) and a pre-configured `Dockerfile.serverless` that ensures CUDA correctly builds without re-downloading massive Torch binaries.

### Deploying to RunPod:
1. Fork or clone this repository to your RunPod environment or a container registry.
2. Build the Docker image using the provided `Dockerfile.serverless`:
   ```bash
   docker build -t chatterbox-serverless -f Dockerfile.serverless .
   ```
3. Push to your registry and create a Serverless endpoint in RunPod.
4. Test the endpoint locally before deploying:
   ```shell
   uv run scripts/local_test.py --model_type turbo --text "Testing the API"
   ```

---

## 📊 Comprehensive Benchmarks

Curious about the actual speedups? We provide a robust benchmarking suite that tests Chunking overhead, Real-Time Factor (RTF) across CUDA Graphs vs Standard, Whisper Validation latency, and Long-Text scaling.

👉 **[View the full benchmark results and comparisons here](BENCHMARK.md)**

We use extremely difficult texts (Alice in Wonderland, heavy acronyms/numbers) to prove the pipeline's robustness.

```shell
uv run scripts/benchmark.py
```

---

## 🙏 Acknowledgements

This repository heavily builds upon the groundbreaking work of others.
* **[Resemble AI](https://github.com/resemble-ai/chatterbox):** Creators of the original open-source Chatterbox models.
* **[alexandrainst/coral_chatterbox](https://github.com/alexandrainst/coral_chatterbox):** Source of the CUDA Graph fast-path and PyTorch SDPA modernizations.
* **[petermg/Chatterbox-TTS-Extended](https://github.com/petermg/Chatterbox-TTS-Extended):** Inspiration for the rich Gradio web application.
* **[rsxdalv/chatterbox](https://github.com/rsxdalv/chatterbox):** Baseline for the fast inference optimizations.

---
*Built with ❤️ for the open-source AI community.*
