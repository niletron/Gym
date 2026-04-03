"""Tests for data format converters."""

import json
import tempfile
from pathlib import Path

from slime_integration.data_converter import (
    convert_file,
    nemogym_to_slime,
    slime_to_nemogym,
)


class TestSlimeToNemoGym:
    def test_simple_string_prompt(self):
        row = {"input": "What is 2+2?", "label": "4"}
        result = slime_to_nemogym(row)
        assert result["responses_create_params"]["input"][0]["role"] == "user"
        assert result["responses_create_params"]["input"][0]["content"][0]["text"] == "What is 2+2?"
        assert result["verifier_metadata"]["label"] == "4"

    def test_conversation_prompt(self):
        row = {
            "input": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "label": "4",
        }
        result = slime_to_nemogym(row)
        assert len(result["responses_create_params"]["input"]) == 2
        assert result["responses_create_params"]["input"][0]["role"] == "system"

    def test_with_system_prompt(self):
        row = {"input": "What is 2+2?", "label": "4"}
        result = slime_to_nemogym(row, system_prompt="You are helpful.")
        assert len(result["responses_create_params"]["input"]) == 2
        assert result["responses_create_params"]["input"][0]["role"] == "system"

    def test_with_metadata(self):
        row = {"input": "test", "label": "answer", "metadata": {"rm_type": "math"}}
        result = slime_to_nemogym(row)
        assert result["verifier_metadata"]["rm_type"] == "math"
        assert result["verifier_metadata"]["label"] == "answer"

    def test_custom_keys(self):
        row = {"prompt": "test", "answer": "result", "meta": {"key": "val"}}
        result = slime_to_nemogym(row, input_key="prompt", label_key="answer", metadata_key="meta")
        assert result["responses_create_params"]["input"][0]["content"][0]["text"] == "test"
        assert result["verifier_metadata"]["label"] == "result"


class TestNemoGymToSlime:
    def test_simple_conversion(self):
        row = {
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "type": "message",
                        "content": [{"type": "input_text", "text": "What is 2+2?"}],
                    }
                ]
            },
            "verifier_metadata": {"label": "4"},
        }
        result = nemogym_to_slime(row)
        assert result["input"] == "What is 2+2?"
        assert result["label"] == "4"

    def test_multi_message_conversion(self):
        row = {
            "responses_create_params": {
                "input": [
                    {"role": "system", "type": "message", "content": [{"type": "input_text", "text": "Be helpful"}]},
                    {"role": "user", "type": "message", "content": [{"type": "input_text", "text": "Hello"}]},
                ]
            },
            "verifier_metadata": {"label": "Hi"},
        }
        result = nemogym_to_slime(row)
        assert isinstance(result["input"], list)
        assert len(result["input"]) == 2


class TestFileConversion:
    def test_roundtrip(self, tmp_path):
        # Create Slime JSONL
        slime_data = [
            {"input": "What is 2+2?", "label": "4", "metadata": {"rm_type": "math"}},
            {"input": "What is 3+3?", "label": "6"},
        ]
        slime_path = tmp_path / "slime.jsonl"
        nemogym_path = tmp_path / "nemogym.jsonl"
        roundtrip_path = tmp_path / "roundtrip.jsonl"

        with open(slime_path, "w") as f:
            for row in slime_data:
                f.write(json.dumps(row) + "\n")

        # Slime -> NeMo-Gym
        count = convert_file(str(slime_path), str(nemogym_path), direction="slime_to_nemogym")
        assert count == 2

        # NeMo-Gym -> Slime
        count = convert_file(str(nemogym_path), str(roundtrip_path), direction="nemogym_to_slime")
        assert count == 2

        # Verify roundtrip
        with open(roundtrip_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert lines[0]["input"] == "What is 2+2?"
        assert lines[0]["label"] == "4"
