# 🏗️ Chatterbox Studio - System Architecture

This document outlines the high-level architecture of this fork, detailing how requests flow from the user interfaces down to the hardware-accelerated PyTorch models.

## 1. High-Level Flow

The system is designed with strict separation of concerns, divided into three main layers: **Client Entrypoints**, the **Pipeline Wrapper**, and the **Raw PyTorch Models**.

```mermaid
flowchart TD
    %% Define Styles
    classDef client fill:#3b82f6,stroke:#1e40af,stroke-width:2px,color:#fff
    classDef wrapper fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
    classDef model fill:#8b5cf6,stroke:#5b21b6,stroke-width:2px,color:#fff
    classDef whisper fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff

    subgraph ClientLayer ["1. Client Layer (Entrypoints)"]
        A["Gradio Studio (app.py)"]:::client
        B["RunPod Serverless (rp_handler.py)"]:::client
        C["Local Scripts (quickstart.py)"]:::client
    end

    subgraph WrapperLayer ["2. Inference Pipeline (chatterbox_serverless)"]
        D["ChatterboxInference Wrapper"]:::wrapper
        E["NLTK Sentence Chunking & Text Packing"]:::wrapper
        F["Parallel Whisper Validation"]:::whisper
    end

    subgraph ModelLayer ["3. Raw PyTorch Models (chatterbox)"]
        G["ChatterboxTurboTTS (CUDA Graphs)"]:::model
        H["ChatterboxMultilingualTTS"]:::model
    end

    %% Connections
    A --> D
    B --> D
    C --> D

    D --> E
    E -->|"Chunk 1...N"| G
    E -->|"Chunk 1...N"| H
    G -->|"Candidate Audio"| F
    H -->|"Candidate Audio"| F
    F -->|"Lowest WER Selection"| D
```

---

## 2. Layer Breakdown

### Layer 1: Client Entrypoints (`examples/app.py`, `rp_handler.py`)
This layer handles User Interfaces and API connections. It intentionally contains **zero logic** regarding text splitting, tensor manipulation, or validation. 
* **`app.py`:** The unified Gradio Studio. Uses a lazy-loaded Singleton pattern to hot-swap models without memory leaks.
* **`rp_handler.py`:** The RunPod serverless worker. Parses incoming JSON requests, decodes base64 audio prompts, and routes them directly to the pipeline.

### Layer 2: The Inference Pipeline (`src/chatterbox_serverless`)
This is the core "brain" of the fork. The `ChatterboxInference` class wraps the underlying PyTorch models and intercepts raw text inputs to provide robust preprocessing.
* **Text Normalization:** Converts tricky numbers and symbols into speakable words.
* **Chunking (`splitter.py`):** Uses NLTK's `sent_tokenize` to safely split paragraphs at natural boundaries, then repacks them into dense strings up to a `max_chunk_chars` limit to minimize fixed-cost inference overhead.
* **Whisper Validation (`inference.py` / `validation.py`):** Utilizes a thread-safe `ThreadPoolExecutor` to generate multiple variations of a single audio chunk and score them with `faster-whisper`. It then seamlessly stitches the lowest-WER chunks together using cross-fades and silence padding.

### Layer 3: Raw PyTorch Models (`src/chatterbox`)
This is the lowest-level hardware layer. The code here is heavily optimized for execution speed.
* **Bucketed CUDA Graphs:** Rather than dynamically executing the autoregressive loop in PyTorch, the generation sequence is traced and compiled into static C++ CUDA graphs, entirely eliminating CPU bottlenecks.
* **Flash Attention:** Upgraded to use PyTorch 2.1+ `sdpa_kernel` for memory-efficient `MultiHeadAttention`.

---

## 3. The `num_candidates` Retry Flow
A critical architectural addition in this fork is the hallucination prevention loop.

```mermaid
sequenceDiagram
    participant User as Client
    participant Pipe as ChatterboxInference
    participant Model as TTS Model
    participant Whisper as Faster-Whisper

    User->>Pipe: generate_fast(text, num_candidates=3)
    Pipe->>Pipe: Chunk text into Sentences (NLTK)
    loop For each Sentence Chunk
        Pipe->>Model: Request 3 independent audio variations
        Model-->>Pipe: Return [Audio_A, Audio_B, Audio_C]
        
        par Whisper Validation Loop
            Pipe->>Whisper: Transcribe Audio_A
            Pipe->>Whisper: Transcribe Audio_B
            Pipe->>Whisper: Transcribe Audio_C
        end
        
        Whisper-->>Pipe: Return [WER_A, WER_B, WER_C]
        Pipe->>Pipe: Discard high WER. Keep lowest WER audio.
    end
    Pipe->>Pipe: Stitch chosen audio chunks together
    Pipe-->>User: Return single, hallucination-free WAV
```
