import os
import sys

# Ensure the src/ directory is in the Python path so local imports work correctly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import time
import torch
import torchaudio as ta
from pathlib import Path
from rich.console import Console
from rich.table import Table

from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox_serverless.inference import ChatterboxInference
from chatterbox.utils.splitter import chunk_text, split_sentences

console = Console()
OUTPUT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_RUNS = 3

def format_runs(times):
    import numpy as np
    return f"{np.mean(times):.2f} ± {np.std(times):.2f}"

def benchmark_chunking_strategies():
    console.rule("[bold magenta]Benchmarking: Chunking Strategies (NLTK vs Packed)[/bold magenta]")
    
    text = "Here is a short sentence. And another one. Followed by a slightly longer sentence that might normally be its own chunk. But with the new packing logic, these should be grouped together into a single chunk up to the maximum character limit. This significantly reduces the overhead of invoking the TTS model multiple times for very short fragments, which is a common issue with raw sentence splitting. Let's see how they compare."
    
    console.print("Raw NLTK (split_sentences):")
    nltk_chunks = split_sentences(text)
    for i, c in enumerate(nltk_chunks):
        console.print(f"  Chunk {i+1}: {len(c)} chars")
        
    console.print("\nPacked (chunk_text):")
    packed_chunks = chunk_text(text)
    for i, c in enumerate(packed_chunks):
        console.print(f"  Chunk {i+1}: {len(c)} chars")
        
    table = Table(title="Chunking Comparison")
    table.add_column("Method", style="cyan")
    table.add_column("Number of Chunks", justify="right", style="magenta")
    table.add_column("Max Chunk Length", justify="right", style="green")
    table.add_column("Avg Chunk Length", justify="right", style="yellow")
    
    import numpy as np
    table.add_row("Raw NLTK", str(len(nltk_chunks)), str(max(len(c) for c in nltk_chunks)), f"{np.mean([len(c) for c in nltk_chunks]):.1f}")
    table.add_row("Packed", str(len(packed_chunks)), str(max(len(c) for c in packed_chunks)), f"{np.mean([len(c) for c in packed_chunks]):.1f}")
    
    console.print(table)

def benchmark_fast_vs_normal():
    console.rule("[bold red]Benchmarking: Normal vs Fast (CUDA Graphs)[/bold red]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        console.print("[yellow]Note: Benchmarking on CPU. CUDA graphs (fast path) only provides speedups on CUDA devices. Comparing anyway.[/yellow]")
        
    console.print("Loading ChatterboxTurboTTS...")
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    
    # Warmup
    console.print("Warming up models...")
    dummy_text = "Hello world, this is a short test."
    _ = model.generate(dummy_text)
    if hasattr(model, "generate_fast"):
        _ = model.generate_fast(dummy_text)
    
    text = "The quick brown fox jumps over the lazy dog. Generating audio locally using autoregressive transformer models requires extreme optimization."
    
    times_normal = []
    rtfs_normal = []
    
    times_fast = []
    rtfs_fast = []
    
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        wav_normal = model.generate(text)
        t = time.perf_counter() - start
        audio_dur = wav_normal.shape[1] / model.sr
        times_normal.append(t)
        rtfs_normal.append(t / audio_dur)
        
        if hasattr(model, "generate_fast"):
            start = time.perf_counter()
            wav_fast = model.generate_fast(text)
            t = time.perf_counter() - start
            audio_dur = wav_fast.shape[1] / model.sr
            times_fast.append(t)
            rtfs_fast.append(t / audio_dur)

    ta.save(OUTPUT_DIR / "bench_normal.wav", wav_normal, model.sr)
    if hasattr(model, "generate_fast"):
        ta.save(OUTPUT_DIR / "bench_fast.wav", wav_fast, model.sr)
    
    table = Table(title=f"Inference Speed Comparison ({NUM_RUNS} runs)")
    table.add_column("Method", justify="right", style="cyan")
    table.add_column("Generation Time (s)", justify="right", style="magenta")
    table.add_column("RTF (lower is better)", justify="right", style="yellow")
    
    table.add_row("Standard generate()", format_runs(times_normal), format_runs(rtfs_normal))
    if times_fast:
        table.add_row("CUDA Graphs generate_fast()", format_runs(times_fast), format_runs(rtfs_fast))
        import numpy as np
        speedup = np.mean(times_normal) / np.mean(times_fast)
        console.print(f"[bold green]Speedup Multiplier: {speedup:.2f}x[/bold green]")
    
    console.print(table)

def benchmark_whisper_validation():
    console.rule("[bold blue]Benchmarking: Without vs With Whisper Validation[/bold blue]")
    
    try:
        import faster_whisper
    except ImportError:
        console.print("[red]faster-whisper is not installed. To run this benchmark, install it with:[/red] pip install faster-whisper")
        return
        
    console.print("Loading ChatterboxInference (Pipeline)...")
    from chatterbox_serverless.validation import validate_audio
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ChatterboxInference.from_pretrained(model_type="turbo", device=device)
    
    tricky_text = "In 1999, Dr. J.R.R. Tolkien's friend, Mr. O'Connor, paid $4,592.33 for a bizarre, antique artifact... wasn't it?"
    
    console.print("Warming up Whisper model...")
    _ = pipeline.generate_fast("Warmup", num_candidates=1)
    
    times_unval = []
    times_val = []
    
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        wav_unval = pipeline.generate_fast(tricky_text, num_candidates=1)
        times_unval.append(time.perf_counter() - start)
        
        start = time.perf_counter()
        wav_val = pipeline.generate_fast(tricky_text, num_candidates=3)
        times_val.append(time.perf_counter() - start)
        
    sr = getattr(pipeline.model, "sr", 24000)
    wav_unval_np = wav_unval.squeeze().cpu().numpy()
    val_unval = validate_audio(wav_unval_np, sr, tricky_text, backend="faster-whisper", language="en")
    wer_unval = val_unval.get("wer", "N/A")
    
    wav_val_np = wav_val.squeeze().cpu().numpy()
    val_val = validate_audio(wav_val_np, sr, tricky_text, backend="faster-whisper", language="en")
    wer_val = val_val.get("wer", "N/A")
    
    ta.save(OUTPUT_DIR / "bench_unval.wav", wav_unval, sr)
    ta.save(OUTPUT_DIR / "bench_val.wav", wav_val, sr)
    
    table = Table(title=f"Whisper Validation Overhead ({NUM_RUNS} runs)")
    table.add_column("Method", justify="right", style="cyan")
    table.add_column("Time Taken (s)", justify="right", style="magenta")
    table.add_column("Word Error Rate (WER)", justify="right", style="red")
    
    table.add_row("No Validation (num_candidates=1)", format_runs(times_unval), f"{wer_unval:.2f}" if isinstance(wer_unval, float) else str(wer_unval))
    table.add_row("Parallel Validated (num_candidates=3)", format_runs(times_val), f"{wer_val:.2f}" if isinstance(wer_val, float) else str(wer_val))
    
    console.print(table)
    
def benchmark_long_text():
    console.rule("[bold green]Benchmarking: Long Text[/bold green]")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ChatterboxInference.from_pretrained(model_type="turbo", device=device)
    
    base_text = "This is a sentence. "
    sizes = [1, 5, 10, 20]
    
    table = Table(title="Long Text Generation RTF")
    table.add_column("Number of Sentences", justify="right", style="cyan")
    table.add_column("Generation Time (s)", justify="right", style="magenta")
    table.add_column("Audio Duration (s)", justify="right", style="green")
    table.add_column("RTF", justify="right", style="yellow")
    
    for size in sizes:
        text = base_text * size
        start = time.perf_counter()
        wav = pipeline.generate_fast(text)
        t = time.perf_counter() - start
        dur = wav.shape[1] / pipeline.sr
        table.add_row(str(size), f"{t:.2f}", f"{dur:.2f}", f"{t/dur:.3f}")
        
    console.print(table)

if __name__ == "__main__":
    console.print("\n[bold]Starting Chatterbox Benchmarks...[/bold]\n")
    benchmark_chunking_strategies()
    print("\n")
    benchmark_fast_vs_normal()
    print("\n")
    benchmark_whisper_validation()
    print("\n")
    benchmark_long_text()
    console.print("\n[bold]Benchmarks Complete.[/bold]")
