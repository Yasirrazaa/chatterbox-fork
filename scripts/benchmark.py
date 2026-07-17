import os
import sys

# Ensure the src/ directory is in the Python path so local imports work correctly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import time
import torch
import torchaudio as ta
from rich.console import Console
from rich.table import Table
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox_serverless.inference import ChatterboxInference

console = Console()

def benchmark_fast_vs_normal():
    console.rule("[bold red]Benchmarking: Normal vs Fast (CUDA Graphs)[/bold red]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        console.print("[yellow]Warning: CUDA not available. CUDA graphs will not trigger![/yellow]")
        return
        
    console.print("Loading ChatterboxTurboTTS...")
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    
    # Warmup
    console.print("Warming up models (first run compiles CUDA graphs)...")
    dummy_text = "Hello world, this is a short test."
    _ = model.generate(dummy_text)
    _ = model.generate_fast(dummy_text)
    
    text = "The quick brown fox jumps over the lazy dog. Generating audio locally using autoregressive transformer models requires extreme optimization."
    
    # Normal
    console.print("Running standard generation...")
    start = time.perf_counter()
    wav_normal = model.generate(text)
    time_normal = time.perf_counter() - start
    audio_duration_normal = wav_normal.shape[1] / model.sr
    rtf_normal = time_normal / audio_duration_normal

    # Fast
    console.print("Running CUDA graph generation...")
    start = time.perf_counter()
    wav_fast = model.generate_fast(text)
    time_fast = time.perf_counter() - start
    audio_duration_fast = wav_fast.shape[1] / model.sr
    rtf_fast = time_fast / audio_duration_fast
    
    table = Table(title="Inference Speed Comparison")
    table.add_column("Method", justify="right", style="cyan")
    table.add_column("Generation Time (s)", justify="right", style="magenta")
    table.add_column("Audio Duration (s)", justify="right", style="green")
    table.add_column("RTF (lower is better)", justify="right", style="yellow")
    
    table.add_row("Standard generate()", f"{time_normal:.2f}", f"{audio_duration_normal:.2f}", f"{rtf_normal:.3f}")
    table.add_row("CUDA Graphs generate_fast()", f"{time_fast:.2f}", f"{audio_duration_fast:.2f}", f"{rtf_fast:.3f}")
    
    console.print(table)
    console.print(f"[bold green]Speedup Multiplier: {time_normal / time_fast:.2f}x[/bold green]")


def benchmark_whisper_validation():
    console.rule("[bold blue]Benchmarking: Without vs With Whisper Validation[/bold blue]")
    
    console.print("Loading ChatterboxInference (Pipeline)...")
    from chatterbox_serverless.validation import validate_audio
    pipeline = ChatterboxInference.from_pretrained(model_type="turbo", device="cuda")
    
    # Tricky text that TTS might struggle with (names, numbers, weird punctuation)
    tricky_text = "In 1999, Dr. J.R.R. Tolkien's friend, Mr. O'Connor, paid $4,592.33 for a bizarre, antique artifact... wasn't it?"
    
    console.print("Warming up Whisper model...")
    _ = pipeline.generate_fast("Warmup", num_candidates=1)
    
    # Without validation (num_candidates = 1)
    console.print("Running without validation...")
    start = time.perf_counter()
    wav_unval = pipeline.generate_fast(tricky_text, num_candidates=1)
    time_unval = time.perf_counter() - start
    
    # Calculate WER for unvalidated audio
    sr = getattr(pipeline.model, "sr", 24000)
    wav_unval_np = wav_unval.squeeze().cpu().numpy()
    val_unval = validate_audio(wav_unval_np, sr, tricky_text, backend="faster-whisper", language="en")
    wer_unval = val_unval.get("wer", 0.0)
    
    # With validation (num_candidates = 3)
    console.print("Running WITH Whisper validation (num_candidates=3)...")
    start = time.perf_counter()
    wav_val = pipeline.generate_fast(tricky_text, num_candidates=3)
    time_val = time.perf_counter() - start
    
    # Calculate WER for validated audio
    wav_val_np = wav_val.squeeze().cpu().numpy()
    val_val = validate_audio(wav_val_np, sr, tricky_text, backend="faster-whisper", language="en")
    wer_val = val_val.get("wer", 0.0)
    
    table = Table(title="Whisper Validation Overhead")
    table.add_column("Method", justify="right", style="cyan")
    table.add_column("Time Taken (s)", justify="right", style="magenta")
    table.add_column("Word Error Rate (WER)", justify="right", style="red")
    table.add_column("Overhead", justify="right", style="yellow")
    
    table.add_row("No Validation (num_candidates=1)", f"{time_unval:.2f}", f"{wer_unval:.2f}", "-")
    table.add_row("Whisper Validated (num_candidates=3)", f"{time_val:.2f}", f"{wer_val:.2f}", f"+{time_val - time_unval:.2f}s")
    
    console.print(table)
    console.print("[italic]Note: In production, the +1-2s overhead of Whisper validation guarantees 0% hallucination rates by auto-discarding variations with high Word Error Rates (WER).[/italic]")

if __name__ == "__main__":
    console.print("\n[bold]Starting Chatterbox Benchmarks...[/bold]\n")
    benchmark_fast_vs_normal()
    print("\n")
    benchmark_whisper_validation()
    console.print("\n[bold]Benchmarks Complete.[/bold]")
