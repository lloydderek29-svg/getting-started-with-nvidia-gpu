import argparse

import torch
from transformers import TextStreamer

from gpubook.llm.loading import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("prompt")
    a = ap.parse_args()

    model, tok = load_model(a.model)
    print(f"weights on GPU: "
          f"{model.get_memory_footprint() / 2**30:.2f} GiB")

    messages = [{"role": "user", "content": a.prompt}]
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)

    streamer = TextStreamer(tok, skip_prompt=True)
    with torch.no_grad():
        model.generate(
            **enc, max_new_tokens=a.max_new,
            do_sample=False, temperature=None,
            top_p=None, top_k=None, streamer=streamer)


if __name__ == "__main__":
    main()
