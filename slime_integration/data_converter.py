"""Bidirectional JSONL conversion between Slime and NeMo-Gym data formats.

Slime JSONL (one line)::

    {"input": "What is 2+2?", "label": "4", "metadata": {"rm_type": "math"}}

NeMo-Gym JSONL (one line)::

    {"responses_create_params": {"input": [...]}, "verifier_metadata": {...}}

CLI usage::

    python -m slime_integration.data_converter input.jsonl output.jsonl \\
        --direction slime_to_nemogym

Programmatic usage::

    from slime_integration.data_converter import slime_to_nemogym, nemogym_to_slime
    row = slime_to_nemogym({"input": "2+2?", "label": "4"})
"""

import json
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Slime → NeMo-Gym
# ---------------------------------------------------------------------------


def slime_to_nemogym(
    row: Dict[str, Any],
    input_key: str = "input",
    label_key: str = "label",
    metadata_key: str = "metadata",
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert one Slime JSONL row into NeMo-Gym format."""
    prompt = row.get(input_key, "")
    label = row.get(label_key)
    metadata = row.get(metadata_key, {}) or {}

    input_msgs = _prompt_to_messages(prompt, system_prompt)

    verifier_metadata = dict(metadata)
    if label is not None:
        verifier_metadata["label"] = label
        verifier_metadata["expected_answer"] = label

    return {
        "responses_create_params": {"input": input_msgs, "model": "slime"},
        "verifier_metadata": verifier_metadata,
    }


# ---------------------------------------------------------------------------
# NeMo-Gym → Slime
# ---------------------------------------------------------------------------


def nemogym_to_slime(
    row: Dict[str, Any],
    input_key: str = "input",
    label_key: str = "label",
    metadata_key: str = "metadata",
) -> Dict[str, Any]:
    """Convert one NeMo-Gym JSONL row into Slime format."""
    rcp = row.get("responses_create_params", {})
    vm = row.get("verifier_metadata", {})

    prompt = _messages_to_prompt(rcp.get("input", []))
    label = vm.get("label") or vm.get("expected_answer")
    metadata = {k: v for k, v in vm.items() if k not in ("label", "expected_answer")}

    result: Dict[str, Any] = {input_key: prompt}
    if label is not None:
        result[label_key] = label
    if metadata:
        result[metadata_key] = metadata
    return result


# ---------------------------------------------------------------------------
# File conversion
# ---------------------------------------------------------------------------


def convert_file(
    input_path: str,
    output_path: str,
    direction: str = "slime_to_nemogym",
    input_key: str = "input",
    label_key: str = "label",
    metadata_key: str = "metadata",
    system_prompt: Optional[str] = None,
) -> int:
    """Convert an entire JSONL file. Returns number of rows processed."""
    fn = slime_to_nemogym if direction == "slime_to_nemogym" else nemogym_to_slime
    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            kwargs: Dict[str, Any] = dict(input_key=input_key, label_key=label_key, metadata_key=metadata_key)
            if direction == "slime_to_nemogym":
                kwargs["system_prompt"] = system_prompt
            fout.write(json.dumps(fn(row, **kwargs), ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prompt_to_messages(prompt: Any, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    if system_prompt:
        msgs.append(_msg("system", system_prompt))
    if isinstance(prompt, str):
        msgs.append(_msg("user", prompt))
    elif isinstance(prompt, list):
        for m in prompt:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str):
                content = [{"type": "input_text", "text": content}]
            msgs.append({"role": role, "type": "message", "content": content})
    else:
        msgs.append(_msg("user", str(prompt)))
    return msgs


def _msg(role: str, text: str) -> Dict[str, Any]:
    return {"role": role, "type": "message", "content": [{"type": "input_text", "text": text}]}


def _messages_to_prompt(input_messages):
    """Convert NeMo-Gym input messages back to a Slime prompt."""
    if isinstance(input_messages, str):
        return input_messages
    if not isinstance(input_messages, list):
        return str(input_messages)

    prompt = []
    for msg in input_messages:
        role = msg.get("role", "user")
        parts = msg.get("content", [])
        if isinstance(parts, list):
            text = " ".join(p.get("text", "") for p in parts if p.get("type") in ("input_text", "text"))
        elif isinstance(parts, str):
            text = parts
        else:
            text = str(parts)
        prompt.append({"role": role, "content": text})

    # Simplify single user message to a plain string
    if len(prompt) == 1 and prompt[0]["role"] == "user":
        return prompt[0]["content"]
    return prompt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Convert between Slime and NeMo-Gym JSONL formats")
    p.add_argument("input_path")
    p.add_argument("output_path")
    p.add_argument("--direction", choices=["slime_to_nemogym", "nemogym_to_slime"], default="slime_to_nemogym")
    p.add_argument("--input-key", default="input")
    p.add_argument("--label-key", default="label")
    p.add_argument("--metadata-key", default="metadata")
    p.add_argument("--system-prompt", default=None)
    args = p.parse_args()

    n = convert_file(
        args.input_path, args.output_path,
        direction=args.direction,
        input_key=args.input_key, label_key=args.label_key, metadata_key=args.metadata_key,
        system_prompt=args.system_prompt,
    )
    print(f"Converted {n} rows ({args.direction}): {args.input_path} -> {args.output_path}")
