"""CPU transport smoke: tiny random DHTA model, official renderer/parser/scorer.

Not a benchmark result or training test. Uses one existing evaluation sample.
Optional audio isolation is restricted to this test for the login-node ABI.
"""
import argparse
import json
import importlib.machinery
import os
from pathlib import Path
import sys
import types


def main():
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--datasets", required=True)
    p.add_argument("--work", required=True)
    args = p.parse_args()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "third_party/embodied_eval_snapshot")]
    os.environ["NOTEBOOK_CONTAINER"] = "1"
    audio = types.ModuleType("torchaudio")
    audio.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
    audio.functional = types.SimpleNamespace(rnnt_loss=lambda *a, **k: None)
    sys.modules["torchaudio"] = audio
    print("TINY_SMOKE: importing torch", flush=True)
    import torch
    torch.set_num_threads(4)
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
    from embodied_eval.models.dhta import prepare_dhta_model_class
    print("TINY_SMOKE: loading processor", flush=True)
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=False, local_files_only=True)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ["[2D_QA]", "[TASK_PLAN]"]})
    print("TINY_SMOKE: loading config", flush=True)
    config = AutoConfig.from_pretrained(args.base, trust_remote_code=False, local_files_only=True)
    text = config.text_config
    text.hidden_size = 32
    text.intermediate_size = 64
    # The real architecture is hybrid; its cache requires both layer kinds.
    text.num_hidden_layers = 2
    text.layer_types = ["linear_attention", "full_attention"]
    text.num_attention_heads = 1
    text.num_key_value_heads = 1
    text.linear_num_key_heads = text.linear_num_value_heads = 1
    text.linear_key_head_dim = text.linear_value_head_dim = 8
    text.vocab_size = len(processor.tokenizer)
    vision = config.vision_config
    vision.depth, vision.hidden_size, vision.intermediate_size = 1, 32, 64
    vision.num_heads, vision.out_hidden_size = 1, 32
    values = dict(dhta_enabled=True, dhta_layers=[1], dhta_rank=2, dhta_alpha=2.,
                  dhta_qa_token="[2D_QA]", dhta_task_plan_token="[TASK_PLAN]",
                  dhta_qa_token_id=processor.tokenizer.convert_tokens_to_ids("[2D_QA]"),
                  dhta_task_plan_token_id=processor.tokenizer.convert_tokens_to_ids("[TASK_PLAN]"),
                  dhta_require_route_token=True)
    for target in (config, text):
        for key, value in values.items(): setattr(target, key, value)
    prepare_dhta_model_class(config)
    print("TINY_SMOKE: constructing tiny model", flush=True)
    model = Qwen3_5ForConditionalGeneration(config)
    model.save_pretrained(work / "tiny-model")
    processor.save_pretrained(work / "tiny-model")
    del model
    print("TINY_SMOKE: evaluating fixture", flush=True)
    from abot_er.evaluation import has_response_text, run
    run(argparse.Namespace(model=str(work / "tiny-model"), datasets=args.datasets,
                           output=str(work / "evaluation"), tasks="emspatial-bench",
                           judge_model=None, no_judge=True, batch_size=1, device="cpu",
                           seed=42, limit=1, smoke_max_tokens=4, dry_run=False, check_data=False))
    assert (work / "evaluation/reproduction_manifest.json").is_file()
    sample_files = list((work / "evaluation").rglob("samples_*.json"))
    assert len(sample_files) == 1, "Exactly one sample artifact required"
    records = [json.loads(line) for line in sample_files[0].read_text().splitlines() if line.strip()]
    assert len(records) == 1
    response = records[0].get("resps") or records[0].get("filtered_resps")
    assert has_response_text(response), "Generation failure cannot count as a passed smoke"
    faulthandler.cancel_dump_traceback_later()
    print("EQA_TINY_END_TO_END_FINISHED: synthetic transport validation, not an accuracy result")


if __name__ == "__main__": main()
