import os
import sys
import time
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jsonargparse import CLI
from receval.settings import CLISettings
from receval.tasks import run_lm_eval
from receval.tasks.val_loss import run_val_loss, load_val_texts_from_parquet
from receval.tasks.core_eval import run_core_eval
from receval.tasks.core_extended_eval import run_core_extended_eval



def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs, flush=True)


def sync_device(device):
    if device is None:
        return
    device_type = getattr(device, "type", str(device).split(":")[0])
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def sample_next_token(logits, temperature: float, top_k=None, top_p=None, do_sample: bool = True):
    import torch.nn.functional as F

    logits = logits / max(temperature, 1e-8)
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

    if do_sample:
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return logits.argmax(dim=-1, keepdim=True)


def create_generation_cache(model, input_ids, max_new_tokens: int):
    if not hasattr(model, "create_cache"):
        return None, {}

    batch_size = input_ids.shape[0]
    max_seq_len = input_ids.shape[1] + max_new_tokens
    if hasattr(getattr(model, "config", None), "mean_recurrence"):
        return model.create_cache(num_steps=model.config.mean_recurrence), {"num_steps": model.config.mean_recurrence}
    return model.create_cache(batch_size, max_seq_len=max_seq_len), {}


def forward_generation_step(model, input_ids, cache=None, cache_kwargs=None):
    cache_kwargs = cache_kwargs or {}
    if hasattr(model, "forward_for_generation"):
        if hasattr(getattr(model, "config", None), "mean_recurrence"):
            return model.forward_for_generation(input_ids, past_key_values=cache, **cache_kwargs), cache
        return model.forward_for_generation(input_ids, kv_cache=cache, **cache_kwargs), cache

    outputs = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    return {"logits": outputs.logits, "past_key_values": outputs.past_key_values}, outputs.past_key_values


@torch.no_grad()
def generate_with_timing(model, input_ids, max_new_tokens: int, temperature: float, top_k=None, top_p=None, do_sample: bool = True, device=None):
    if max_new_tokens <= 0:
        return input_ids.clone(), {
            "prefill_tokens": input_ids.numel(),
            "prefill_seconds": 0.0,
            "prefill_tokens_per_second": 0.0,
            "generate_tokens": 0,
            "generate_seconds": 0.0,
            "generate_tokens_per_second": 0.0,
        }

    cache, cache_kwargs = create_generation_cache(model, input_ids, max_new_tokens)
    generated = input_ids.clone()

    sync_device(device)
    prefill_start = time.perf_counter()
    outputs, cache = forward_generation_step(model, input_ids, cache=cache, cache_kwargs=cache_kwargs)
    logits = outputs["logits"][:, -1, :]
    if "past_key_values" in outputs:
        cache = outputs["past_key_values"]
    elif "kv_cache" in outputs:
        cache = outputs["kv_cache"]
    next_token = sample_next_token(logits, temperature, top_k=top_k, top_p=top_p, do_sample=do_sample)
    sync_device(device)
    prefill_seconds = time.perf_counter() - prefill_start
    generated = torch.cat([generated, next_token], dim=-1)

    generate_seconds = 0.0
    generate_tokens = 0
    current_input = next_token
    for _ in range(max_new_tokens - 1):
        sync_device(device)
        generate_start = time.perf_counter()
        outputs, cache = forward_generation_step(model, current_input, cache=cache, cache_kwargs=cache_kwargs)
        logits = outputs["logits"][:, -1, :]
        if "past_key_values" in outputs:
            cache = outputs["past_key_values"]
        elif "kv_cache" in outputs:
            cache = outputs["kv_cache"]
        next_token = sample_next_token(logits, temperature, top_k=top_k, top_p=top_p, do_sample=do_sample)
        sync_device(device)
        generate_seconds += time.perf_counter() - generate_start
        generate_tokens += next_token.numel()
        generated = torch.cat([generated, next_token], dim=-1)
        current_input = next_token

    prefill_tokens = input_ids.numel()
    metrics = {
        "prefill_tokens": prefill_tokens,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": prefill_tokens / prefill_seconds if prefill_seconds > 0 else 0.0,
        "generate_tokens": generate_tokens,
        "generate_seconds": generate_seconds,
        "generate_tokens_per_second": generate_tokens / generate_seconds if generate_seconds > 0 else 0.0,
    }
    return generated, metrics


def print_generation_metrics(metrics):
    generate_tps = metrics["generate_tokens_per_second"]
    generate_tps_text = f"{generate_tps:.2f}" if metrics["generate_tokens"] > 0 else "n/a"
    print0(
        "  "
        f"Prefill: {metrics['prefill_tokens']:,} tokens in {metrics['prefill_seconds']:.4f}s "
        f"({metrics['prefill_tokens_per_second']:.2f} token/s) | "
        f"Generate: {metrics['generate_tokens']:,} tokens in {metrics['generate_seconds']:.4f}s "
        f"({generate_tps_text} token/s)"
    )


def aggregate_generation_metrics(metrics_list):
    prefill_tokens = sum(m["prefill_tokens"] for m in metrics_list)
    prefill_seconds = sum(m["prefill_seconds"] for m in metrics_list)
    generate_tokens = sum(m["generate_tokens"] for m in metrics_list)
    generate_seconds = sum(m["generate_seconds"] for m in metrics_list)
    return {
        "prefill_tokens": prefill_tokens,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": prefill_tokens / prefill_seconds if prefill_seconds > 0 else 0.0,
        "generate_tokens": generate_tokens,
        "generate_seconds": generate_seconds,
        "generate_tokens_per_second": generate_tokens / generate_seconds if generate_seconds > 0 else 0.0,
    }


def setup_distributed(settings: CLISettings):
    if settings.ddp:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(settings.ddp_local_rank)
    if settings.device_type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except AttributeError:
            pass  # Not all PyTorch versions have this


def cleanup_distributed(settings: CLISettings):
    if settings.ddp:
        torch.distributed.destroy_process_group()


def load_model(settings: CLISettings):
    from parcae_lm.tokenizer import Tokenizer

    # HuggingFace transformers model (e.g. GPT-2)
    if settings.hf_path is not None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print0(f"Loading HuggingFace model: {settings.hf_path}")
        model = AutoModelForCausalLM.from_pretrained(settings.hf_path)
        if settings.device is not None:
            model.to(settings.device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(settings.hf_path)
        return model, tokenizer

    # Parcae/GPT model from HuggingFace repo (e.g. SandyResearch/parcae-140m)
    if settings.hf_repo is not None:
        from receval.models.parcae import ModelingParcae
        print0(f"Loading from HuggingFace: {settings.hf_repo}")
        model = ModelingParcae.from_pretrained(settings.hf_repo, device=settings.device)
        model.eval()
        if settings.tokenizer_path:
            tokenizer = Tokenizer(settings.tokenizer_path)
        else:
            tokenizer = Tokenizer.from_pretrained("SandyResearch/parcae-tokenizer")
        return model, tokenizer

    # Local checkpoint
    if settings.model_impl == "gpt":
        from receval.models.gpt import ModelingGPT
        ModelClass = ModelingGPT
    elif settings.model_impl == "parcae":
        from receval.models.parcae import ModelingParcae
        ModelClass = ModelingParcae
    else:
        raise ValueError(f"Unknown model_impl: {settings.model_impl}")

    print0(f"Loading {settings.model_impl} model: {settings.model_name}")
    assert settings.model_config is not None, "model_config required"

    if settings.override_mean_recurrence is not None and hasattr(settings.model_config, 'mean_recurrence'):
        original_recurrence = settings.model_config.mean_recurrence
        settings.model_config.mean_recurrence = settings.override_mean_recurrence
        print0(f"Overriding mean_recurrence: {original_recurrence} -> {settings.override_mean_recurrence}")

    model = ModelClass(settings.model_config)

    if settings.checkpoint_path:
        print0(f"Loading checkpoint: {settings.checkpoint_path}")
        state = torch.load(settings.checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in state:
            model.load_state_dict(state["model"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

    model.to(settings.device)
    model.eval()

    if settings.tokenizer_path:
        tokenizer = Tokenizer(settings.tokenizer_path)
    else:
        raise ValueError("tokenizer_path not found in run config")

    return model, tokenizer


def run_sample_task(model, tokenizer, settings: CLISettings):
    if not settings._is_main_process:
        return {}

    task_settings = settings.tasks.sample
    results = {"conditioned": [], "unconditioned": [], "metrics": []}

    print0("\n" + "=" * 60)
    print0("Sample Generation")
    print0("=" * 60)

    for prompt in task_settings.prompts:
        tokens = tokenizer.encode(prompt, return_tensors=False)
        input_ids = torch.tensor([tokens], device=settings.device)
        with settings._get_autocast_context():
            output, metrics = generate_with_timing(
                model,
                input_ids,
                max_new_tokens=task_settings.max_tokens,
                temperature=task_settings.temperature,
                top_k=task_settings.top_k,
                top_p=task_settings.top_p,
                do_sample=task_settings.temperature > 0,
                device=settings.device,
            )
        text = tokenizer.decode(output[0].tolist())
        print_generation_metrics(metrics)
        print0(f"\n{text}")
        results["conditioned"].append(text)
        results["metrics"].append({"type": "conditioned", "prompt": prompt, **metrics})

    if task_settings.num_unconditioned > 0:
        print0("\nUnconditioned samples:")
        bos = getattr(tokenizer, "bos_id", 1)
        input_ids = torch.tensor([[bos]], device=settings.device)
        for _ in range(task_settings.num_unconditioned):
            with settings._get_autocast_context():
                output, metrics = generate_with_timing(
                    model,
                    input_ids,
                    max_new_tokens=task_settings.max_tokens,
                    temperature=task_settings.temperature,
                    do_sample=True,
                    device=settings.device,
                )
            text = tokenizer.decode(output[0].tolist())
            print_generation_metrics(metrics)
            print0(f"\n{text}")
            results["unconditioned"].append(text)
            results["metrics"].append({"type": "unconditioned", **metrics})

    if results["metrics"]:
        aggregate_metrics = aggregate_generation_metrics(results["metrics"])
        print0("\nSample throughput aggregate:")
        print_generation_metrics(aggregate_metrics)
        results["aggregate_metrics"] = aggregate_metrics

    return results


def main():
    settings: CLISettings = CLI(CLISettings)
    setup_distributed(settings)

    print0("=" * 60)
    print0("Evaluation")
    print0("=" * 60)
    print0(f"Run dir: {settings.out_dir}")
    print0(f"Model: {settings.model_name}")
    print0(f"Checkpoint: {settings.checkpoint_path}")
    print0(f"Tasks: {settings.eval_tasks}")
    print0(f"Device: {settings.device}")
    if settings.override_mean_recurrence is not None:
        print0(f"Override recurrence: {settings.override_mean_recurrence}")
    print0("=" * 60)

    if settings.checkpoint_path is None and settings.hf_path is None and settings.hf_repo is None:
        print0("\nERROR: No checkpoint found!")
        print0("Check that out_dir contains checkpoints or pass --checkpoint_path")
        cleanup_distributed(settings)
        return

    model, tokenizer = load_model(settings)
    all_results = {}

    for task in settings.eval_task_list:
        print0(f"\nRunning task: {task}")
        t0 = time.time()

        if task == "lm_eval":
            results = run_lm_eval(model, tokenizer, settings)
            all_results["lm_eval"] = results.results
            if settings._is_main_process:
                print0(f"\nLM-Eval Results:")
                for task_name, task_result in results.task_scores.items():
                    print0(f"  {task_name}:")
                    if isinstance(task_result, dict):
                        for metric, value in task_result.items():
                            if isinstance(value, (int, float)):
                                print0(f"    {metric}: {value:.4f}")
                    else:
                        print0(f"    {task_result}")

        elif task == "sample":
            results = run_sample_task(model, tokenizer, settings)
            all_results["sample"] = results

        elif task == "bpb":
            val_data_dir = settings.tasks.bpb.val_data_dir
            if not val_data_dir:
                print0("WARNING: No val_data_dir specified for BPB task, skipping")
                all_results["bpb"] = {}
            else:
                print0(f"Loading validation data from: {val_data_dir}")
                val_texts = load_val_texts_from_parquet(
                    val_data_dir,
                    max_files=settings.tasks.bpb.max_files
                )
                print0(f"Loaded {len(val_texts)} validation texts")
                results = run_val_loss(
                    model, tokenizer, settings, val_texts,
                    max_samples=settings.tasks.bpb.max_samples
                )
                all_results["bpb"] = {
                    "loss": results.loss,
                    "perplexity": results.perplexity,
                    "bits_per_byte": results.bits_per_byte,
                    "num_tokens": results.num_tokens,
                    "num_bytes": results.num_bytes,
                }
                if settings._is_main_process:
                    print0(f"\nValidation Results:")
                    print0(f"  Loss: {results.loss:.4f}")
                    print0(f"  Perplexity: {results.perplexity:.2f}")
                    print0(f"  Bits per byte: {results.bits_per_byte:.4f}")
                    print0(f"  Tokens evaluated: {results.num_tokens:,}")
                    print0(f"  Bytes evaluated: {results.num_bytes:,}")

        elif task == "core":
            max_seq_len = settings.sequence_length or 2048
            seeds = settings.tasks.core.seeds
            with settings._get_autocast_context():
                results = run_core_eval(
                    model, tokenizer, settings.device,
                    max_seq_len=max_seq_len,
                    max_per_task=settings.tasks.core.max_per_task,
                    seeds=seeds
                )
            all_results["core"] = results
            if settings._is_main_process:
                print0(f"\nCORE Results:")
                # Check if we have aggregated results (multiple seeds)
                if 'aggregated' in results:
                    for task_name in results["results"].keys():
                        acc = results["aggregated"]["results"][task_name]
                        acc_std = results["aggregated"]["results_std"][task_name]
                        centered = results["aggregated"]["centered_results"][task_name]
                        centered_std = results["aggregated"]["centered_results_std"][task_name]
                        print0(f"  {task_name}: acc={acc:.4f}±{acc_std:.4f} centered={centered:.4f}±{centered_std:.4f}")
                    print0(f"\n  CORE metric: {results['core_metric']:.4f} ± {results['aggregated']['core_metric_std']:.4f}")
                else:
                    for task_name, acc in results["results"].items():
                        centered = results["centered_results"][task_name]
                        print0(f"  {task_name}: acc={acc:.4f} centered={centered:.4f}")
                    print0(f"\n  CORE metric: {results['core_metric']:.4f}")

        elif task == "core_extended":
            max_seq_len = settings.sequence_length or 2048
            seeds = settings.tasks.core_extended.seeds
            with settings._get_autocast_context():
                results = run_core_extended_eval(
                    model, tokenizer, settings.device,
                    max_seq_len=max_seq_len,
                    max_per_task=settings.tasks.core_extended.max_per_task,
                    seeds=seeds
                )
            all_results["core_extended"] = results
            if settings._is_main_process:
                print0(f"\nCORE Extended Results:")
                if 'aggregated' in results:
                    for task_name in results["results"].keys():
                        acc = results["aggregated"]["results"][task_name]
                        acc_std = results["aggregated"]["results_std"][task_name]
                        centered = results["aggregated"]["centered_results"][task_name]
                        centered_std = results["aggregated"]["centered_results_std"][task_name]
                        print0(f"  {task_name}: acc={acc:.4f}±{acc_std:.4f} centered={centered:.4f}±{centered_std:.4f}")
                    print0(f"\n  CORE metric: {results['core_metric']:.4f} ± {results['aggregated']['core_metric_std']:.4f}")
                    print0(f"  CORE Extended metric: {results['core_extended_metric']:.4f} ± {results['aggregated']['core_extended_metric_std']:.4f}")
                else:
                    for task_name, acc in results["results"].items():
                        centered = results["centered_results"][task_name]
                        print0(f"  {task_name}: acc={acc:.4f} centered={centered:.4f}")
                    print0(f"\n  CORE metric: {results['core_metric']:.4f}")
                    print0(f"  CORE Extended metric: {results['core_extended_metric']:.4f}")

        print0(f"Task {task} completed in {time.time() - t0:.2f}s")

    if settings._is_main_process and settings.out_dir:
        eval_dir = Path(settings.out_dir) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        output_path = eval_dir / f"{settings.eval_name}.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print0(f"\nResults saved to: {output_path}")

    cleanup_distributed(settings)
    print0("\nEvaluation complete.")


if __name__ == "__main__":
    main()
