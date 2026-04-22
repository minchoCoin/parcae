import argparse
import contextlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(precision: str):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def autocast_context(device: torch.device, dtype):
    if dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def sync_device(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def sample_next_token(logits, temperature: float, top_k=None, top_p=None):
    import torch.nn.functional as F

    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def decode_tokens(tokenizer, token_ids):
    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        except TypeError:
            return tokenizer.decode(token_ids)
    raise TypeError("Tokenizer does not provide decode().")


def get_parcae_special_id(tokenizer, token: str):
    try:
        return tokenizer.encode_special(token)
    except Exception:
        return None


def build_parcae_chat_input(tokenizer, messages, device: torch.device, max_context_tokens: int):
    bos = getattr(tokenizer, "bos_id", None) or 1
    user_start = get_parcae_special_id(tokenizer, "<|user|>")
    user_end = get_parcae_special_id(tokenizer, "<|/user|>")
    assistant_start = get_parcae_special_id(tokenizer, "<|assistant|>")
    assistant_end = get_parcae_special_id(tokenizer, "<|/assistant|>")
    has_chat_tokens = all(token_id is not None for token_id in [user_start, user_end, assistant_start, assistant_end])

    chat_messages = messages[:]
    if chat_messages and chat_messages[0]["role"] == "system":
        system_text = chat_messages[0]["content"]
        chat_messages = chat_messages[1:]
        if chat_messages and chat_messages[0]["role"] == "user":
            chat_messages[0] = {
                "role": "user",
                "content": f"{system_text}\n\n{chat_messages[0]['content']}",
            }

    if not has_chat_tokens:
        lines = []
        for message in chat_messages:
            role = "User" if message["role"] == "user" else "Assistant"
            lines.append(f"{role}: {message['content']}")
        lines.append("Assistant:")
        ids = tokenizer.encode("\n".join(lines), return_tensors=False)
        ids = ids[-max_context_tokens:]
        return torch.tensor([ids], dtype=torch.long, device=device), set()

    ids = [bos]
    for message in chat_messages:
        role = message["role"]
        content_ids = tokenizer.encode(message["content"], return_tensors=False)
        if role == "user":
            ids.extend([user_start])
            ids.extend(content_ids)
            ids.extend([user_end])
        elif role == "assistant":
            ids.extend([assistant_start])
            ids.extend(content_ids)
            ids.extend([assistant_end])

    ids.extend([assistant_start])
    ids = [token_id for token_id in ids if token_id is not None]
    ids = ids[-max_context_tokens:]
    return torch.tensor([ids], dtype=torch.long, device=device), {assistant_end}


def build_hf_chat_input(tokenizer, messages, device: torch.device, max_context_tokens: int):
    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    else:
        lines = []
        for message in messages:
            role = message["role"].capitalize()
            lines.append(f"{role}: {message['content']}")
        lines.append("Assistant:")
        prompt = "\n".join(lines)
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    return input_ids[:, -max_context_tokens:], {tokenizer.eos_token_id}


def forward_parcae_generation(model, input_ids, cache, num_steps):
    outputs = model.forward_for_generation(
        input_ids,
        num_steps=num_steps,
        past_key_values=cache,
    )
    return outputs["logits"][:, -1, :], outputs.get("past_key_values", cache)


@torch.no_grad()
def generate_hf(model, input_ids, tokenizer, args, device: torch.device, dtype):
    if args.max_new_tokens <= 0:
        return "", {
            "prefill_tokens": input_ids.numel(),
            "prefill_seconds": 0.0,
            "generate_tokens": 0,
            "generate_seconds": 0.0,
            "generated_tokens_total": 0,
        }

    generated_tokens = []
    stop_ids = {tokenizer.eos_token_id}

    with autocast_context(device, dtype):
        sync_device(device)
        prefill_start = time.perf_counter()
        outputs = model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        next_token = sample_next_token(logits, args.temperature, args.top_k, args.top_p)
        sync_device(device)
        prefill_seconds = time.perf_counter() - prefill_start

        token_id = int(next_token.item())
        if token_id not in stop_ids:
            generated_tokens.append(token_id)

        decode_seconds = 0.0
        decode_token_count = 0
        current_input = next_token
        for _ in range(args.max_new_tokens - 1):
            if token_id in stop_ids:
                break
            sync_device(device)
            decode_start = time.perf_counter()
            outputs = model(input_ids=current_input, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            next_token = sample_next_token(logits, args.temperature, args.top_k, args.top_p)
            sync_device(device)
            decode_seconds += time.perf_counter() - decode_start

            token_id = int(next_token.item())
            if token_id in stop_ids:
                break
            generated_tokens.append(token_id)
            decode_token_count += 1
            current_input = next_token

    return decode_tokens(tokenizer, generated_tokens), {
        "prefill_tokens": input_ids.numel(),
        "prefill_seconds": prefill_seconds,
        "generate_tokens": decode_token_count,
        "generate_seconds": decode_seconds,
        "generated_tokens_total": len(generated_tokens),
    }


@torch.no_grad()
def generate_parcae(model, input_ids, tokenizer, stop_ids, args, device: torch.device, dtype):
    if args.max_new_tokens <= 0:
        return "", {
            "prefill_tokens": input_ids.numel(),
            "prefill_seconds": 0.0,
            "generate_tokens": 0,
            "generate_seconds": 0.0,
            "generated_tokens_total": 0,
        }

    generated_tokens = []
    stop_ids = {token_id for token_id in stop_ids if token_id is not None}
    eos_id = getattr(tokenizer, "eos_id", None)
    if eos_id is not None:
        stop_ids.add(eos_id)

    with autocast_context(device, dtype):
        num_steps = model.config.mean_recurrence
        cache = model.create_cache(num_steps=num_steps)

        sync_device(device)
        prefill_start = time.perf_counter()
        logits, cache = forward_parcae_generation(model, input_ids, cache, num_steps)
        next_token = sample_next_token(logits, args.temperature, args.top_k, args.top_p)
        sync_device(device)
        prefill_seconds = time.perf_counter() - prefill_start

        token_id = int(next_token.item())
        if token_id not in stop_ids:
            generated_tokens.append(token_id)

        decode_seconds = 0.0
        decode_token_count = 0
        current_input = next_token
        for _ in range(args.max_new_tokens - 1):
            if token_id in stop_ids:
                break
            sync_device(device)
            decode_start = time.perf_counter()
            logits, cache = forward_parcae_generation(model, current_input, cache, num_steps)
            next_token = sample_next_token(logits, args.temperature, args.top_k, args.top_p)
            sync_device(device)
            decode_seconds += time.perf_counter() - decode_start

            token_id = int(next_token.item())
            if token_id in stop_ids:
                break
            generated_tokens.append(token_id)
            decode_token_count += 1
            current_input = next_token

    return decode_tokens(tokenizer, generated_tokens), {
        "prefill_tokens": input_ids.numel(),
        "prefill_seconds": prefill_seconds,
        "generate_tokens": decode_token_count,
        "generate_seconds": decode_seconds,
        "generated_tokens_total": len(generated_tokens),
    }


@torch.no_grad()
def generate_parcae_full_context(model, input_ids, tokenizer, stop_ids, args, device: torch.device, dtype):
    if args.max_new_tokens <= 0:
        return "", {
            "prefill_tokens": input_ids.numel(),
            "prefill_seconds": 0.0,
            "generate_tokens": 0,
            "generate_seconds": 0.0,
            "generated_tokens_total": 0,
        }

    generated = input_ids.clone()
    generated_tokens = []
    stop_ids = {token_id for token_id in stop_ids if token_id is not None}
    eos_id = getattr(tokenizer, "eos_id", None)
    if eos_id is not None:
        stop_ids.add(eos_id)

    with autocast_context(device, dtype):
        sync_device(device)
        prefill_start = time.perf_counter()
        outputs = model(generated, return_logits=True)
        logits = outputs["logits"][:, -1, :]
        next_token = sample_next_token(logits, args.temperature, args.top_k, args.top_p)
        sync_device(device)
        prefill_seconds = time.perf_counter() - prefill_start

        token_id = int(next_token.item())
        if token_id not in stop_ids:
            generated_tokens.append(token_id)
            generated = torch.cat([generated, next_token], dim=-1)

        decode_seconds = 0.0
        decode_token_count = 0
        for _ in range(args.max_new_tokens - 1):
            if token_id in stop_ids:
                break
            sync_device(device)
            decode_start = time.perf_counter()
            outputs = model(generated, return_logits=True)
            logits = outputs["logits"][:, -1, :]
            next_token = sample_next_token(logits, args.temperature, args.top_k, args.top_p)
            sync_device(device)
            decode_seconds += time.perf_counter() - decode_start

            token_id = int(next_token.item())
            if token_id in stop_ids:
                break
            generated_tokens.append(token_id)
            decode_token_count += 1
            generated = torch.cat([generated, next_token], dim=-1)

    return decode_tokens(tokenizer, generated_tokens), {
        "prefill_tokens": input_ids.numel(),
        "prefill_seconds": prefill_seconds,
        "generate_tokens": decode_token_count,
        "generate_seconds": decode_seconds,
        "generated_tokens_total": len(generated_tokens),
    }


def print_metrics(metrics):
    prefill_tps = metrics["prefill_tokens"] / metrics["prefill_seconds"] if metrics["prefill_seconds"] > 0 else 0.0
    generate_tps = metrics["generate_tokens"] / metrics["generate_seconds"] if metrics["generate_seconds"] > 0 else 0.0
    generate_text = f"{generate_tps:.2f}" if metrics["generate_tokens"] > 0 else "n/a"
    print(
        f"prefill token/s: {prefill_tps:.2f} "
        f"({metrics['prefill_tokens']} tokens / {metrics['prefill_seconds']:.4f}s)"
    )
    print(
        f"generation token/s: {generate_text} "
        f"({metrics['generate_tokens']} measured decode tokens / {metrics['generate_seconds']:.4f}s, "
        f"{metrics['generated_tokens_total']} generated total)"
    )


def load_parcae(args, device: torch.device, dtype):
    from parcae_lm.tokenizer import Tokenizer
    from receval.models.parcae import ModelingParcae

    model = ModelingParcae.from_pretrained(args.model, device=device, dtype=dtype)
    model.eval()
    tokenizer_name = args.tokenizer or "SandyResearch/parcae-tokenizer"
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


def load_hf(args, device: torch.device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model if args.tokenizer is None else args.tokenizer)
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive chat eval with prefill and generation token/s.")
    parser.add_argument("--backend", choices=["auto", "parcae", "hf"], default="auto")
    parser.add_argument("--model", default="SandyResearch/parcae-140m", help="HF repo id or local HF model path.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer repo/path. Defaults to model for HF, parcae-tokenizer for Parcae.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--system", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    dtype = get_dtype(args.precision)
    if device.type != "cuda" and args.precision != "fp32":
        dtype = None

    backend = args.backend
    if backend == "auto":
        backend = "parcae" if "parcae" in args.model.lower() else "hf"

    if backend == "parcae":
        model, tokenizer = load_parcae(args, device, dtype)
    else:
        model, tokenizer = load_hf(args, device, dtype)

    print(f"Loaded {backend} model: {args.model} on {device}")
    print("Type /exit or /quit to stop.")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    while True:
        try:
            user_text = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.lower() in {"/exit", "/quit"}:
            break
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        if backend == "parcae":
            input_ids, stop_ids = build_parcae_chat_input(tokenizer, messages, device, args.max_context_tokens)
            try:
                answer, metrics = generate_parcae(model, input_ids, tokenizer, stop_ids, args, device, dtype)
            except Exception as exc:
                print(f"Parcae cache generation failed, retrying without cache: {exc}")
                answer, metrics = generate_parcae_full_context(model, input_ids, tokenizer, stop_ids, args, device, dtype)
        else:
            input_ids, _ = build_hf_chat_input(tokenizer, messages, device, args.max_context_tokens)
            answer, metrics = generate_hf(model, input_ids, tokenizer, args, device, dtype)

        print(f"\nAssistant: {answer}")
        print_metrics(metrics)
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
