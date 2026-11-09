"""Run standalone Hub inference or the installed GS1 runtime."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="bigai/GS1-0.8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtime", choices=["hf", "gs1"], default="hf")
    parser.add_argument("--request", type=Path, default=Path(__file__).with_name("request.json"))
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    precision = "bf16" if args.device.startswith("cuda") else "fp32"
    if args.runtime == "hf":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(args.device).eval()
        result = model.decide(**request, precision=precision)
    else:
        from gs1 import Engine

        engine = Engine.from_pretrained(args.model, device=args.device, precision=precision)
        try:
            result = engine.evaluate(request).model_dump(mode="json")
        finally:
            engine.close()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
