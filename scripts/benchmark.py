"""Benchmark standard vs fast (torch.compile) generation.

Usage:
    python benchmark.py --model_type multilingual --text "Your benchmark sentence here."
    python benchmark.py --runs 3 --use_fast

Reports per-run and average real-time factor (RTF = audio_seconds / wall_seconds).
Higher RTF = faster. On a compatible model (rsxdalv/coral fork) with --use_fast,
you should see a 2-4x speedup versus standard generation.

The base chatterbox-tts==0.1.7 pip package does NOT expose the compile hook, so
--use_fast will log a warning and fall back to standard speed (RTF equal).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rp_handler import handler, initialize_model


def run_once(text: str, model_type: str, language: str, use_fast: bool,
             compile_dtype: str | None, runs: int):
    initialize_model(model_type=model_type, language=language, use_fast=use_fast,
                     compile_dtype=compile_dtype)
    rtf_list = []
    dur_list = []
    for i in range(runs):
        job = {"input": {
            "model_config": {
                "model_type": model_type, "language": language,
                "use_fast": use_fast, "compile_dtype": compile_dtype,
            },
            "text": text, "language_id": language,
            "temperature": 0.8, "cfg_weight": 0.5, "exaggeration": 0.5,
            "output_format": "wav",
        }}
        t0 = time.time()
        res = handler(job)
        elapsed = time.time() - t0
        if res.get("status") != "success":
            print(f"  run {i+1}: ERROR {res.get('error')}")
            continue
        md = res["metadata"]
        rtf = md["realtime_factor"]
        dur = md["duration_seconds"]
        rtf_list.append(rtf)
        dur_list.append(dur)
        print(f"  run {i+1}: RTF={rtf}  audio={dur}s  wall={elapsed:.2f}s")
    return rtf_list, dur_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", default="multilingual",
                    choices=["base", "multilingual", "turbo"])
    ap.add_argument("--language", default="en")
    ap.add_argument("--text", default=(
        "The quick brown fox jumps over the lazy dog. "
        "Chatterbox is an open source text to speech model with voice cloning. "
        "Benchmarking helps us compare standard and optimized inference paths."))
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--compile_dtype", default=None, help="e.g. bfloat16")
    args = ap.parse_args()

    print(f"\n### STANDARD generation ({args.model_type}) ###")
    std_rtf, std_dur = run_once(args.text, args.model_type, args.language,
                                use_fast=False, compile_dtype=None, runs=args.runs)

    print(f"\n### FAST generation ({args.model_type}, use_fast=True) ###")
    fast_rtf, fast_dur = run_once(args.text, args.model_type, args.language,
                                  use_fast=True, compile_dtype=args.compile_dtype,
                                  runs=args.runs)

    if std_rtf and fast_rtf:
        std_avg = sum(std_rtf) / len(std_rtf)
        fast_avg = sum(fast_rtf) / len(fast_rtf)
        speedup = fast_avg / std_avg if std_avg else float('nan')
        print("\n=== SUMMARY ===")
        print(f"  standard avg RTF : {std_avg:.3f}")
        print(f"  fast    avg RTF : {fast_avg:.3f}")
        print(f"  speedup (fast/std): {speedup:.2f}x")
        if speedup < 1.05:
            print("  NOTE: fast path did not speed up — the installed model "
                  "likely lacks the torch.compile hook. Use a fork that adds "
                  "_step_compilation_target (e.g. rsxdalv/chatterbox:fast).")
    else:
        print("\nBenchmark incomplete (some runs errored).")


if __name__ == "__main__":
    main()
