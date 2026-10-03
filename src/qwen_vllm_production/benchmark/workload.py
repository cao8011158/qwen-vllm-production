"""Deterministic synthetic chat, token-sized with a locally available tokenizer."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Prompt:
    index: int
    text: str
    prompt_tokens: int
    sha256: str


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def workload_hash(prompts: list[Prompt]) -> str:
    return text_hash("\n".join(prompt.sha256 for prompt in prompts))


def load_local_tokenizer(path: str) -> Any:
    """Never download a tokenizer during a benchmark."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False,
    )


def build_workload(
    tokenizer: Any, *, count: int, seed: int = 42,
    target_input_tokens: int = 1000, target_output_tokens: int = 256,
    tokenizer_path: str = "", tolerance_tokens: int = 16,
) -> dict:
    if count <= 0 or target_input_tokens <= 0 or target_output_tokens <= 0:
        raise ValueError("Workload counts and token targets must be positive.")
    rng = random.Random(seed)
    topics = ("delivery", "support", "planning", "inventory", "training", "documentation")
    actions = ("review", "schedule", "explain", "compare", "organize", "summarize")
    prompts = []
    for index in range(count):
        # A distinct early identifier limits long shared prefixes across requests.
        prefix = f"Conversation {rng.getrandbits(64):016x}. Please summarize this planning note. "
        words: list[str] = []
        while len(words) < target_input_tokens * 2:
            words.extend(
                f"The team will {rng.choice(actions)} the {rng.choice(topics)} "
                f"plan for item {rng.randrange(10000)} and discuss the next steps.".split()
            )

        def render(word_count: int) -> tuple[str, int]:
            messages = [
                {"role": "system", "content": "You are a concise and helpful planning assistant."},
                {"role": "user", "content": prefix + " ".join(words[:word_count])},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            if not isinstance(text, str):
                raise ValueError("Tokenizer must provide a string chat template.")
            return text, len(tokenizer.encode(text, add_special_tokens=False))

        low, high = 0, len(words)
        best_text, best_count = render(0)
        while low <= high:
            middle = (low + high) // 2
            text, token_count = render(middle)
            if abs(token_count - target_input_tokens) < abs(best_count - target_input_tokens):
                best_text, best_count = text, token_count
            if token_count < target_input_tokens:
                low = middle + 1
            else:
                high = middle - 1
        if abs(best_count - target_input_tokens) > tolerance_tokens:
            raise ValueError(
                f"Cannot fit prompt to {target_input_tokens} +/- {tolerance_tokens} tokens; "
                f"nearest length is {best_count}."
            )
        prompts.append(Prompt(index, best_text, best_count, text_hash(best_text)))
    return {
        "seed": seed, "tokenizer_path": tokenizer_path,
        "target_input_tokens": target_input_tokens, "target_output_tokens": target_output_tokens,
        "input_token_tolerance": tolerance_tokens,
        "chat_template_applied_locally": True, "enable_thinking": False,
        "workload_sha256": workload_hash(prompts),
        "prompts": [asdict(prompt) for prompt in prompts],
    }


def validate_workload(
    payload: dict, tokenizer: Any, *, count: int, seed: int,
    target_input_tokens: int, target_output_tokens: int,
) -> list[Prompt]:
    if payload.get("chat_template_applied_locally") is not True or payload.get("enable_thinking") is not False:
        raise ValueError("Saved workload must use the same local chat template and thinking policy.")
    for key, expected in (
        ("seed", seed), ("target_input_tokens", target_input_tokens),
        ("target_output_tokens", target_output_tokens),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"Saved workload {key} differs from the requested protocol.")
    prompts = [Prompt(**record) for record in payload["prompts"]]
    if len(prompts) != count:
        raise ValueError("Saved workload count must equal warmup + measured requests.")
    for index, prompt in enumerate(prompts):
        if prompt.index != index or prompt.sha256 != text_hash(prompt.text):
            raise ValueError("Saved workload order or prompt hash is invalid.")
        if prompt.prompt_tokens != len(tokenizer.encode(prompt.text, add_special_tokens=False)):
            raise ValueError("Saved workload tokenizer differs from the current tokenizer.")
        if abs(prompt.prompt_tokens - target_input_tokens) > payload.get("input_token_tolerance", 16):
            raise ValueError("Saved prompt is outside the declared input token tolerance.")
    if payload.get("workload_sha256") != workload_hash(prompts):
        raise ValueError("Saved workload digest is invalid.")
    return prompts
