"""Local testing for the RunPod serverless handler.

Run the worker handler locally without the RunPod API:

    python local_test.py                 # uses test_input.json
    python local_test.py --input test_input_long.json --out outputs/local.wav
    python local_test.py --model_type turbo --text "Hello from the turbo model!"

This mimics RunPod's local testing flow: it builds a ``{"input": ...}`` job
dict and calls ``handler(job)`` directly, then writes the returned audio to disk.

For the real RunPod local test (handler invoked by the worker), see:
https://docs.runpod.io/serverless/development/local-testing
"""

import argparse
import base64
import json
import os
import sys

# Ensure the project root and src/ directory are in the Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from rp_handler import handler, initialize_model


def build_job(args) -> dict:
    """Build a job dict from CLI args or a JSON file."""
    if args.input:
        with open(args.input) as f:
            data = json.load(f)
        return data if "input" in data else {"input": data}

    # Build a minimal job from CLI flags
    model_config = {
        "model_type": args.model_type,
        "language": args.language,
        "normalize_text": True,
        "sentence_split": True,
        "inter_sentence_silence_ms": 100,
        "use_fast": args.use_fast,
        "compile_dtype": args.compile_dtype,
    }
    inp = {
        "model_config": model_config,
        "text": args.text,
        "language_id": args.language,
        "temperature": 0.8,
        "cfg_weight": 0.5,
        "exaggeration": 0.5,
        "output_format": args.output_format,
    }
    if args.preprocess:
        inp["preprocess_text"] = True
    if args.denoise:
        inp["denoise"] = True
    if args.normalize:
        inp["normalize"] = True
    if args.validate:
        inp["validate"] = True
    return {"input": inp}


def main():
    ap = argparse.ArgumentParser(description="Local test for rp_handler")
    ap.add_argument("--input", help="Path to a JSON job file (default: tests/serverless/test_input.json if no args provided)")
    ap.add_argument("--out", default="outputs/local_test_output.wav",
                    help="Output audio path")
    ap.add_argument("--model_type", default="multilingual",
                    choices=["base", "multilingual", "turbo"])
    ap.add_argument("--language", default="en")
    ap.add_argument("--text", default="Hello world! This is a local test of the Chatterbox serverless worker.")
    ap.add_argument("--output_format", default="wav", choices=["wav", "mp3", "flac"])
    ap.add_argument("--use_fast", action="store_true", help="Try torch.compile fast path")
    ap.add_argument("--compile_dtype", default=None, help="e.g. bfloat16")
    ap.add_argument("--preprocess", action="store_true")
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    # If no arguments were provided at all, default to the test JSON file
    if len(sys.argv) == 1:
        args.input = "tests/serverless/test_input.json"

    if args.input and not os.path.exists(args.input):
        print(f"Warning: Test input not found at {args.input}")

    job = build_job(args)

    # Pre-initialize the model once (cached for subsequent calls)
    initialize_model()

    print("\n=== Running handler locally ===")
    result = handler(job)
    print(json.dumps(result.get("metadata", {}), indent=2))

    if result.get("status") == "success":
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        audio_b64 = result["audio_base64"]
        with open(args.out, "wb") as f:
            f.write(base64.b64decode(audio_b64))
        print(f"\nAudio written to: {args.out} "
              f"({len(base64.b64decode(audio_b64))} bytes)")
    else:
        print(f"\nHandler returned error: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
