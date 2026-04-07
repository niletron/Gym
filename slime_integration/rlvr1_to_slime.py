"""Convert Nemotron RLVR1 dataset from NeMo-Gym format to Slime flat JSONL.

Usage::

    python -m slime_integration.rlvr1_to_slime \\
        --input data/rlvr1_filled/rlvr1.jsonl \\
        --output data/rlvr1_slime.jsonl

Produces one JSONL where every line has::

    {"messages": [...], "label": str|None, "metadata": {...}, "tools": [...]|None}

The ``metadata.env_type`` field tells the routing RM which NeMo-Gym resources
server to dispatch each sample to.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset → env_type mapping
# ---------------------------------------------------------------------------

DATASET_TO_ENV: Dict[str, str] = {
    "super_v3_lcsft_step1000_tau_pivot": "single_step_tool_use",
    "super_v3_lcsft_step1000_instruction_following": "instruction_following",
    "super_v3_lcsft_step1000_comp_coding": "code_gen",
    "super_v3_lcsft_step1000_skyworks": "math",
    "super_v3_lcsft_step1000_dapo17k": "math",
    "super_v3_lcsft_step1000_stem_mcqa": "mcqa",
    "super_v3_lcsft_step1000_structured_outputs": "structured_outputs",
    "super_v3_lcsft_step1000_calendar_v2": "calendar",
    "super_v3_lcsft_step1000_reasoning_gym": "reasoning_gym",
    "super_v3_lcsft_step1000_lean": "math_formal_lean",
    "super_v3_lcsft_step1000_workbench": "workplace_assistant",
}

# Datasets to skip (GenRM, judge-dependent, SWE sandbox, etc.)
SKIP_DATASETS = {
    "hs3",
    "hs4_20260106_combinedrubricsonly",
    "lmarena_5k",
    "super_identity_w_principle_genrm",
    "safety_v0.3.0",
    "super_v3_lcsft_step1000_multichallenge_vanilla_and_advanced_len40k",
    "super_v3_lcsft_step1000_jailbreak_and_overrefusal",
    "super_v3_lcsft_step1000_single_step_swe_swegym_and_scale",
}

# ---------------------------------------------------------------------------
# Message format conversion
# ---------------------------------------------------------------------------


def _flatten_content(content: Any) -> str:
    """Convert NeMo-Gym message content to plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("input_text", "output_text", "text"):
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _convert_messages(rcp_input: List[Dict]) -> List[Dict[str, str]]:
    """Convert NeMo-Gym input messages to OpenAI chat format."""
    messages = []
    for msg in rcp_input:
        role = msg.get("role", "user")
        content = _flatten_content(msg.get("content", ""))

        # NeMo-Gym uses special types for tool calls/outputs
        msg_type = msg.get("type", "")
        if msg_type == "function_call":
            # Tool call message — include as assistant with function info
            name = msg.get("name", "")
            arguments = msg.get("arguments", "")
            content = json.dumps({"name": name, "arguments": arguments})
            role = "assistant"
        elif msg_type == "function_call_output":
            content = msg.get("output", content)
            role = "tool"

        if content:
            messages.append({"role": role, "content": content})
    return messages


def _extract_tools(rcp: Dict) -> Optional[List]:
    """Extract tools from responses_create_params if present."""
    tools = rcp.get("tools")
    if tools and isinstance(tools, list) and len(tools) > 0:
        return tools
    return None


# ---------------------------------------------------------------------------
# Per-environment metadata extraction
# ---------------------------------------------------------------------------


def _extract_env_metadata(record: Dict, env_type: str) -> tuple[Optional[str], Dict[str, Any]]:
    """Extract (label, metadata) for a given environment type.

    Returns (label, metadata_dict). The metadata_dict always includes env_type.
    """
    meta: Dict[str, Any] = {"env_type": env_type}

    if env_type == "single_step_tool_use":
        meta["expected_action"] = record.get("expected_action")
        return None, meta

    if env_type == "instruction_following":
        for key in ("instruction_id_list", "kwargs", "grading_mode", "prompt", "id"):
            if key in record:
                meta[key] = record[key]
        return None, meta

    if env_type == "code_gen":
        vm = record.get("verifier_metadata", {})
        meta["verifier_metadata"] = vm
        return None, meta

    if env_type == "math":
        label = record.get("expected_answer")
        meta["question"] = record.get("question", "")
        meta["expected_answer"] = label
        return str(label) if label is not None else None, meta

    if env_type == "mcqa":
        label = record.get("expected_answer")
        for key in ("expected_answer", "options", "grading_mode", "template_metadata", "uuid"):
            if key in record:
                meta[key] = record[key]
        return str(label) if label is not None else None, meta

    if env_type == "structured_outputs":
        for key in ("schema_str", "schema_type", "schema_fields_count"):
            if key in record:
                meta[key] = record[key]
        return None, meta

    if env_type == "calendar":
        meta["exp_cal_state"] = record.get("exp_cal_state")
        return None, meta

    if env_type == "reasoning_gym":
        label = record.get("answer")
        meta["question"] = record.get("question", "")
        meta["answer"] = label
        # reasoning_gym has nested metadata with source_dataset
        rg_meta = record.get("metadata", {})
        meta["reasoning_gym_metadata"] = rg_meta
        return str(label) if label is not None else None, meta

    if env_type == "math_formal_lean":
        for key in ("header", "formal_statement", "informal_prefix", "name"):
            if key in record:
                meta[key] = record[key]
        return None, meta

    if env_type == "workplace_assistant":
        for key in ("ground_truth", "id", "category", "environment_name"):
            if key in record:
                meta[key] = record[key]
        return None, meta

    return None, meta


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert_record(record: Dict) -> Optional[Dict]:
    """Convert one RLVR1 record to Slime format.

    Returns None for skipped records.
    """
    dataset = record.get("dataset", "")

    if dataset in SKIP_DATASETS:
        return None

    env_type = DATASET_TO_ENV.get(dataset)
    if env_type is None:
        return None

    rcp = record.get("responses_create_params", {})
    rcp_input = rcp.get("input", [])

    # Skip records with empty input (unresolved placeholders)
    if not rcp_input:
        return None

    messages = _convert_messages(rcp_input)
    if not messages:
        return None

    tools = _extract_tools(rcp)
    label, metadata = _extract_env_metadata(record, env_type)

    result: Dict[str, Any] = {
        "messages": messages,
        "label": label,
        "metadata": metadata,
    }
    if tools:
        result["tools"] = tools

    return result


def convert_file(input_path: str, output_path: str) -> Dict[str, int]:
    """Convert full RLVR1 JSONL file. Returns per-env counts."""
    import collections

    counts = collections.Counter()
    skipped = 0
    errors = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for i, line in enumerate(fin):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            converted = convert_record(record)
            if converted is None:
                skipped += 1
                continue

            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            env_type = converted["metadata"]["env_type"]
            counts[env_type] += 1

            if (i + 1) % 10000 == 0:
                logger.info("Processed %d lines (%d converted, %d skipped)", i + 1, sum(counts.values()), skipped)

    total = sum(counts.values())
    logger.info("Done: %d converted, %d skipped, %d errors", total, skipped, errors)
    for env, c in counts.most_common():
        logger.info("  %s: %d (%.1f%%)", env, c, c / total * 100)

    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description="Convert RLVR1 to Slime format")
    parser.add_argument("--input", required=True, help="Input RLVR1 JSONL")
    parser.add_argument("--output", required=True, help="Output Slime JSONL")
    parser.add_argument("--sample", type=int, default=0, help="Sample N per env_type (0=all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.sample > 0:
        # Two-pass: first convert all, then sample
        import collections
        import random

        random.seed(42)
        logger.info("Converting with sampling: %d per env_type", args.sample)

        # Pass 1: collect all records grouped by env_type
        groups: Dict[str, List[str]] = collections.defaultdict(list)
        with open(args.input) as fin:
            for line in fin:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                converted = convert_record(record)
                if converted is None:
                    continue
                env_type = converted["metadata"]["env_type"]
                groups[env_type].append(json.dumps(converted, ensure_ascii=False))

        # Pass 2: sample and write
        with open(args.output, "w") as fout:
            total = 0
            for env_type, records in sorted(groups.items()):
                sampled = random.sample(records, min(args.sample, len(records)))
                for r in sampled:
                    fout.write(r + "\n")
                    total += 1
                logger.info("  %s: %d/%d sampled", env_type, len(sampled), len(records))
            logger.info("Total sampled: %d", total)
    else:
        convert_file(args.input, args.output)


if __name__ == "__main__":
    main()
