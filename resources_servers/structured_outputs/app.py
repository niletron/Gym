# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ---------------------------------------------------------------------------
# structured_outputs runs untrusted JSON schemas against untrusted model
# output through openapi_schema_validator. Pathological schemas (deep
# recursion, patternProperties with backtracking regex) can hang the
# validator. We isolate verify() in a subprocess pool that SIGKILLs on
# timeout so one bad sample can't freeze the server.
# ---------------------------------------------------------------------------

import json
import re
from enum import StrEnum
from typing import Any, ClassVar, Dict

import xmltodict
import yaml
from fastapi import FastAPI
from openapi_schema_validator import validate as validate_against_schema_openapi

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.grader_pool import GraderPool


# ---------------------------------------------------------------------------
# Worker-side functions — must be module-level for spawn pickle.
# ---------------------------------------------------------------------------


def _strictify_schema(schema: Any) -> None:
    if isinstance(schema, dict):
        if "properties" in schema:
            schema["required"] = list(schema["properties"])
            schema["additionalProperties"] = False
        for v in schema.values():
            _strictify_schema(v)


def _coerce_xml_types(data: Any, schema: Dict[str, Any]) -> Any:
    """Same behavior as the class method; duplicated here so spawn workers
    don't need to pickle the server instance."""
    if not isinstance(schema, dict) or "type" not in schema:
        return data

    schema_type = schema["type"]

    if schema_type == "object" and isinstance(data, dict):
        properties = schema.get("properties", {})
        coerced = {}
        for key, value in data.items():
            if key in properties:
                coerced[key] = _coerce_xml_types(value, properties[key])
            else:
                coerced[key] = value
        return coerced

    if schema_type == "array":
        items_schema = schema.get("items", {})
        if isinstance(data, dict) and len(data) == 1:
            data = next(iter(data.values()))
        if not isinstance(data, list):
            data = [data] if data is not None else []
        return [_coerce_xml_types(item, items_schema) for item in data]

    if data is None and schema_type == "string":
        return ""

    if isinstance(data, str):
        try:
            if schema_type == "integer":
                return int(data)
            if schema_type == "number":
                return float(data)
            if schema_type == "boolean":
                lower = data.lower()
                if lower in ("true", "1"):
                    return True
                if lower in ("false", "0"):
                    return False
        except (ValueError, AttributeError):
            pass

    return data


def _parse_content(schema_type: str, content: str):
    st = schema_type.lower()
    if st == "json":
        return json.loads(content)
    if st == "yaml":
        return yaml.safe_load(content)
    if st == "xml":
        return xmltodict.parse(content)
    return None


def _evaluate_in_worker(
    schema_type: str, schema_str: str, response_text: str, xml_coerce_types: bool
) -> float:
    """Runs in a worker process. Returns 0.0 on any grading failure."""
    try:
        schema = json.loads(schema_str)
        _strictify_schema(schema)
        response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
        response_text = re.sub(r"<\|[^|]*\|>", "", response_text).strip()
        response_text = re.sub(r"^```(?:\w*)\s*\n?", "", response_text)
        response_text = re.sub(r"\n?```\s*$", "", response_text)
        response_obj = _parse_content(schema_type, response_text)
        if schema_type.lower() == "xml" and xml_coerce_types:
            response_obj = _coerce_xml_types(response_obj, schema)
        validate_against_schema_openapi(response_obj, schema)
        return 1.0
    except BaseException:
        return 0.0


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class StructuredOutputsResourcesServerConfig(BaseResourcesServerConfig):
    xml_coerce_types: bool = True


class SchemaType(StrEnum):
    JSON = "json"
    YAML = "yaml"
    XML = "xml"


class StructuredOutputsVerifyRequest(BaseVerifyRequest):
    # string representation of schema. For JSON, it is a json dictionary.
    schema_str: str
    schema_type: SchemaType


class StructuredOutputsVerifyResponse(BaseVerifyResponse):
    schema_str: str
    schema_type: SchemaType


class StructuredOutputsResourcesServer(SimpleResourcesServer):
    config: StructuredOutputsResourcesServerConfig

    _POOL: ClassVar[GraderPool] = GraderPool(
        name="structured_outputs",
        timeout_s_env="STRUCTURED_OUTPUTS_VERIFY_TIMEOUT_S",
        workers_env="STRUCTURED_OUTPUTS_POOL_WORKERS",
        default_timeout_s=4.0,  # routing_rm timeout is 5s
        default_workers=4,
    )

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._POOL.bind(self)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        @app.get("/health")
        async def health():
            """End-to-end probe: grade a canned trivial JSON schema."""
            schema_str = '{"type": "object", "properties": {"x": {"type": "integer"}}}'
            response = '{"x": 1}'
            result, reason = await self._POOL.run_or_zero(
                self, _evaluate_in_worker, "json", schema_str, response, False
            )
            return {"status": "ok" if reason == "ok" else reason, "reward": result}

        return app

    async def verify(self, body: StructuredOutputsVerifyRequest) -> StructuredOutputsVerifyResponse:
        schema_type = body.schema_type
        schema_str = body.schema_str

        if schema_type not in list(SchemaType):
            raise NotImplementedError(f"SchemaType must be one of {list(SchemaType)}, got {schema_type} !")

        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue
            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue
                assistant_responses.append(content_item.text)
        response_text = "".join(assistant_responses)

        # Dispatch to subprocess pool — isolates pathological schemas or
        # untrusted XML/YAML parsers from the event loop.
        reward, _reason = await self._POOL.run_or_zero(
            self,
            _evaluate_in_worker,
            str(schema_type.value),
            schema_str,
            response_text,
            self.config.xml_coerce_types,
        )
        return StructuredOutputsVerifyResponse(**body.model_dump(), reward=reward)

    # Legacy methods kept for backwards-compat with anything that imports them
    # directly (tests, tooling).
    def parse_content(self, schema_type: SchemaType, content: str):
        return _parse_content(str(schema_type.value), content)

    def strictify_schema(self, schema: Dict[str, Any]):
        _strictify_schema(schema)

    def coerce_xml_types(self, data: Any, schema: Dict[str, Any]) -> Any:
        return _coerce_xml_types(data, schema)

    def evaluate_structured_output_response(
        self, schema_type: SchemaType, schema_str: str, response_text: str
    ) -> float:
        return _evaluate_in_worker(
            str(schema_type.value), schema_str, response_text, self.config.xml_coerce_types
        )


if __name__ == "__main__":
    StructuredOutputsResourcesServer.run_webserver()
