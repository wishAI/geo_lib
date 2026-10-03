"""Tiny random CPU dependency regression; never motion/inference evidence."""
import json
import os
import sys
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
                  HF_HOME=str(OUT / "encoder_hf_cache"), TEXT_ENCODER_DEVICE="cpu")
sys.path.insert(0, str(OUT / "vendor/kimodo"))

import importlib.metadata as metadata
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, PreTrainedTokenizerFast
from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder

torch.set_num_threads(2)
torch.manual_seed(123)
config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                     num_hidden_layers=1, num_attention_heads=2,
                     num_key_value_heads=2, max_position_embeddings=512)
config._attn_implementation = "eager"
model = LlamaBiModel(config).eval()
results = {}
for dtype in (torch.float32, torch.bfloat16):
    model = model.to(dtype)
    for label, mask in (("visible", torch.ones(1, 4, dtype=torch.long)),
                        ("padding", torch.tensor([[1, 1, 1, 0]]))):
        with torch.no_grad():
            a = model(input_ids=torch.tensor([[1, 2, 3, 4]]), attention_mask=mask,
                      use_cache=False).last_hidden_state
            b = model(input_ids=torch.tensor([[1, 2, 3, 9]]), attention_mask=mask,
                      use_cache=False).last_hidden_state
        delta = float((a[:, 0] - b[:, 0]).abs().max())
        assert delta > .01 if label == "visible" else delta == 0
        results[f"{dtype}_{label}_prefix_delta"] = delta

with tempfile.TemporaryDirectory(dir=OUT, prefix="encoder_tiny_") as tmp:
    tmp = Path(tmp)
    base, mntp, supervised = (tmp / n for n in ("base", "mntp", "supervised"))
    model.float().save_pretrained(base)
    for destination in (mntp, supervised):
        fresh = LlamaBiModel.from_pretrained(base, local_files_only=True)
        adapted = get_peft_model(fresh, LoraConfig(r=2, lora_alpha=4,
                                                 target_modules=["q_proj", "v_proj"]))
        for name, value in adapted.named_parameters():
            if "lora_" in name:
                torch.nn.init.uniform_(value, -.1, .1)
        adapted.save_pretrained(destination, save_embedding_layers=False)
        path = destination / "adapter_config.json"
        obj = json.loads(path.read_text())
        obj["base_model_name_or_path"] = str(base)
        path.write_text(json.dumps(obj))
    config.save_pretrained(mntp)
    path = mntp / "config.json"
    obj = json.loads(path.read_text())
    obj["_name_or_path"] = "meta-llama/Meta-Llama-3-8B-Instruct"
    path.write_text(json.dumps(obj))
    vocabulary = {word: i for i, word in enumerate(
        ["<unk>", "<s>", "</s>", "A", "person", "walks", "waves", ".", "user"])}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>",
                                  bos_token="<s>", eos_token="</s>")
    fast.model_input_names = ["input_ids", "attention_mask"]
    fast.save_pretrained(mntp)
    tok_path = mntp / "tokenizer_config.json"
    tok_config = json.loads(tok_path.read_text())
    tok_config["model_input_names"] = ["input_ids", "attention_mask"]
    tok_path.write_text(json.dumps(tok_config))
    encoder = LLM2VecEncoder(str(mntp), str(supervised), "bfloat16", 16, device="cpu")
    embeddings, lengths = encoder(["A person walks.", "A person waves."])
    assert embeddings.shape == (2, 1, 16) and lengths == [1, 1]
    assert torch.isfinite(embeddings).all()
    assert encoder.model.model.config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert "<|start_header_id|>user<|end_header_id|>" in encoder.model.prepare_for_tokenization("test")
    results["official_wrapper"] = {"shape": list(embeddings.shape), "lengths": lengths,
                                    "dtype": str(embeddings.dtype), "finite": True,
                                    "instruction_framing_preserved": True}
    dtype_counts = {}
    for param in encoder.model.parameters():
        key = str(param.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + param.numel()
    results["official_wrapper"]["parameter_count_by_dtype"] = dtype_counts

report = {
    "scope": "Tiny random CPU fixtures only; no pretrained weights, network, authentication or GPU.",
    "versions": {name: metadata.version(name) for name in
                 ("transformers", "tokenizers", "peft", "accelerate", "huggingface_hub", "torch")},
    "dependency_isolation": "encoder_venv contains encoder overrides; .pth reads other packages from existing task venv without modifying it.",
    "official_dependency_evidence": "https://github.com/McGill-NLP/llm2vec/blob/6bbd52528bee4936786ff0e9eb8a569698b1c731/setup.py",
    "official_dependency_range": "transformers>=4.43.1,<=4.44.2",
    "masking": "Unmodified pinned Kimodo LlamaBiModel, ordinary 2D mask; no mask substitutions or hooks.",
    "results": results,
    "command": "PYTHONDONTWRITEBYTECODE=1 algorithms/motion_anim_generate/outputs/encoder_venv/bin/python algorithms/motion_anim_generate/encoder_check.py",
}
(OUT / "encoder_compatibility_result.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
