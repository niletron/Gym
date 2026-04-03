"""Data format converters between Slime and NeMo-Gym JSONL formats.

Slime JSONL format:
    {"input": "...", "label": "...", "metadata": {...}}

NeMo-Gym JSONL format:
    {"responses_create_params": {"input": [...]}, "verifier_metadata": {...}}
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def slime_to_nemogym(
    slime_row: Dict[str, Any],
    input_key: str = "input",
    label_key: str = "label",
    metadata_key: str = "metadata",
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a single Slime JSONL row to NeMo-Gym format.

    Args:
        slime_row: Dict from Slime JSONL (has input_key, label_key, metadata_key fields)
        input_key: Key for the prompt field in Slime data
        label_key: Key for the ground truth label
        metadata_key: Key for metadata
        system_prompt: Optional system prompt to prepend

    Returns:
        Dict in NeMo-Gym JSONL format
    """
    prompt = slime_row.get(input_key, "")
    label = slime_row.get(label_key)
    metadata = slime_row.get(metadata_key, {}) or {}

    # Build NeMo-Gym input messages
    input_messages = []
    if system_prompt:
        input_messages.append(
            {
                "role": "system",
                "type": "message",
                "content": [{"type": "input_text", "text": system_prompt}],
            }
        )

    if isinstance(prompt, str):
        input_messages.append(
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": prompt}],
            }
        )
    elif isinstance(prompt, list):
        # Already in conversation format
        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                content = [{"type": "input_text", "text": content}]
            input_messages.append(
                {
                    "role": role,
                    "type": "message",
                    "content": content,
                }
            )

    # Build verifier metadata
    verifier_metadata = dict(metadata)
    if label is not None:
        verifier_metadata["label"] = label
        verifier_metadata["expected_answer"] = label

    return {
        "responses_create_params": {
            "input": input_messages,
            "model": "slime",
        },
        "verifier_metadata": verifier_metadata,
    }


def nemogym_to_slime(
    nemogym_row: Dict[str, Any],
    input_key: str = "input",
    label_key: str = "label",
    metadata_key: str = "metadata",
) -> Dict[str, Any]:
    """Convert a single NeMo-Gym JSONL row to Slime format.

    Args:
        nemogym_row: Dict from NeMo-Gym JSONL
        input_key: Key for the prompt field in Slime data
        label_key: Key for the ground truth label
        metadata_key: Key for metadata

    Returns:
        Dict in Slime JSONL format
    """
    rcp = nemogym_row.get("responses_create_params", {})
    verifier_metadata = nemogym_row.get("verifier_metadata", {})

    # Extract prompt from NeMo-Gym input messages
    input_messages = rcp.get("input", [])
    if isinstance(input_messages, str):
        prompt = input_messages
    elif isinstance(input_messages, list):
        # Convert to conversation format
        prompt = []
        for msg in input_messages:
            role = msg.get("role", "user")
            content_parts = msg.get("content", [])
            if isinstance(content_parts, list):
                text = " ".join(p.get("text", "") for p in content_parts if p.get("type") in ("input_text", "text"))
            elif isinstance(content_parts, str):
                text = content_parts
            else:
                text = str(content_parts)
            prompt.append({"role": role, "content": text})

        # If single user message, simplify to string
        if len(prompt) == 1 and prompt[0]["role"] == "user":
            prompt = prompt[0]["content"]
    else:
        prompt = str(input_messages)

    label = verifier_metadata.get("label") or verifier_metadata.get("expected_answer")
    metadata = {k: v for k, v in verifier_metadata.items() if k not in ("label", "expected_answer")}

    result = {input_key: prompt}
    if label is not None:
        result[label_key] = label
    if metadata:
        result[metadata_key] = metadata

    return result


def convert_file(
    input_path: str,
    output_path: str,
    direction: str = "slime_to_nemogym",
    input_key: str = "input",
    label_key: str = "label",
    metadata_key: str = "metadata",
    system_prompt: Optional[str] = None,
) -> int:
    """Convert an entire JSONL file between formats.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        direction: "slime_to_nemogym" or "nemogym_to_slime"
        input_key: Slime's prompt field key
        label_key: Slime's label field key
        metadata_key: Slime's metadata field key
        system_prompt: Optional system prompt (only for slime_to_nemogym)

    Returns:
        Number of rows converted
    """
    converter = slime_to_nemogym if direction == "slime_to_nemogym" else nemogym_to_slime
    count = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            if direction == "slime_to_nemogym":
                converted = converter(
                    row,
                    input_key=input_key,
                    label_key=label_key,
                    metadata_key=metadata_key,
                    system_prompt=system_prompt,
                )
            else:
                converted = converter(
                    row,
                    input_key=input_key,
                    label_key=label_key,
                    metadata_key=metadata_key,
                )

            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1

    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert between Slime and NeMo-Gym JSONL formats")
    parser.add_argument("input_path", help="Path to input JSONL file")
    parser.add_argument("output_path", help="Path to output JSONL file")
    parser.add_argument(
        "--direction",
        choices=["slime_to_nemogym", "nemogym_to_slime"],
        default="slime_to_nemogym",
    )
    parser.add_argument("--input-key", default="input")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--metadata-key", default="metadata")
    parser.add_argument("--system-prompt", default=None)

    args = parser.parse_args()
    count = convert_file(
        args.input_path,
        args.output_path,
        direction=args.direction,
        input_key=args.input_key,
        label_key=args.label_key,
        metadata_key=args.metadata_key,
        system_prompt=args.system_prompt,
    )
    print(f"Converted {count} rows ({args.direction}): {args.input_path} -> {args.output_path}")
