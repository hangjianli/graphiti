"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import typing
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .config import DEFAULT_MAX_TOKENS, LLMConfig
from .openai_base_client import DEFAULT_REASONING, DEFAULT_VERBOSITY, BaseOpenAIClient


class OpenAIClient(BaseOpenAIClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the BaseOpenAIClient and provides OpenAI-specific implementation
    for creating completions.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning: str = DEFAULT_REASONING,
        verbosity: str = DEFAULT_VERBOSITY,
    ):
        """
        Initialize the OpenAIClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
        """
        super().__init__(config, cache, max_tokens, reasoning, verbosity)

        if config is None:
            config = LLMConfig()

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """Create a structured completion using custom parse implementation."""
        return await self._custom_parse(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=response_model,
            reasoning=reasoning,
            verbosity=verbosity,
        )

    async def _custom_parse(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None = None,
        verbosity: str | None = None,
    ) -> Any:
        """Custom parse implementation to replace client.responses.parse API."""
        # Reasoning models (gpt-5 family) don't support temperature
        is_reasoning_model = model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3')
        
        # Prepare messages with additional instructions for structured output
        enhanced_messages = list(messages)
        
        # Add schema instruction to the system message or create one
        schema_instruction = self._build_schema_instruction(response_model, reasoning, verbosity)
        
        # Find system message or add one
        system_message_found = False
        for i, msg in enumerate(enhanced_messages):
            if msg.get('role') == 'system':
                enhanced_messages[i] = {
                    'role': 'system',
                    'content': f"{msg['content']}\n\n{schema_instruction}"
                }
                system_message_found = True
                break
        
        if not system_message_found:
            enhanced_messages.insert(0, {
                'role': 'system',
                'content': schema_instruction
            })
        
        # Create completion with structured output
        # Use max_completion_tokens for GPT-5 models, max_tokens for others
        completion_params = {
            "model": model,
            "messages": enhanced_messages,
        }
        
        # Handle temperature parameter - GPT-5 models don't support null temperature
        if not is_reasoning_model:
            completion_params["temperature"] = temperature
        elif "gpt-5" in model.lower():
            # GPT-5 models only support default temperature (1) or no temperature parameter
            pass  # Don't include temperature parameter at all
        else:
            completion_params["temperature"] = None
        
        # GPT-5 models use max_completion_tokens, others use max_tokens
        if "gpt-5" in model.lower():
            completion_params["max_completion_tokens"] = max_tokens
        else:
            completion_params["max_tokens"] = max_tokens
            
        # Get the schema and ensure additionalProperties is set to false
        schema = response_model.model_json_schema()
        
        # Recursively ensure all objects have additionalProperties: false and required fields
        def ensure_schema_compliance(obj):
            if isinstance(obj, dict):
                if obj.get('type') == 'object' or 'properties' in obj:
                    # Ensure additionalProperties is set to false
                    obj['additionalProperties'] = False
                    
                    # For strict mode, required must include ALL properties
                    if 'properties' in obj:
                        all_properties = list(obj['properties'].keys())
                        obj['required'] = all_properties
                
                for value in obj.values():
                    ensure_schema_compliance(value)
            elif isinstance(obj, list):
                for item in obj:
                    ensure_schema_compliance(item)
        
        ensure_schema_compliance(schema)
        
        response = await self.client.chat.completions.create(**completion_params,
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': response_model.__name__,
                    'schema': schema,
                    'strict': True
                }
            }
        )
        
        # Create a mock response object that matches the expected structure
        return self._create_mock_parse_response(response, response_model)
    
    def _build_schema_instruction(
        self, 
        response_model: type[BaseModel], 
        reasoning: str | None = None, 
        verbosity: str | None = None
    ) -> str:
        """Build instruction text for structured output generation."""
        instruction = f"You must respond with valid JSON that matches this exact schema:\n{json.dumps(response_model.model_json_schema(), indent=2)}"
        
        if reasoning:
            instruction += f"\n\nReasoning effort level: {reasoning}"
        
        if verbosity:
            instruction += f"\nVerbosity level: {verbosity}"
            
        instruction += "\n\nEnsure your response is valid JSON and follows the schema exactly."
        return instruction
    
    def _create_mock_parse_response(self, response: Any, response_model: type[BaseModel]) -> Any:
        """Create a mock response object that matches the expected parse API structure."""
        content = response.choices[0].message.content
        
        # Parse and validate the JSON response
        try:
            parsed_data = json.loads(content) if content else {}
            # Validate against the Pydantic model
            validated_instance = response_model.model_validate(parsed_data)
            
            # Create a mock response object with the expected structure
            class MockParseResponse:
                def __init__(self, output_text: str, validated_data: BaseModel):
                    self.output_text = output_text
                    self.validated_data = validated_data
                    self.refusal = None
            
            return MockParseResponse(content, validated_instance)
            
        except (json.JSONDecodeError, ValueError) as e:
            # Create a mock response with refusal for invalid JSON
            class MockParseRefusalResponse:
                def __init__(self, refusal_reason: str):
                    self.output_text = None
                    self.refusal = refusal_reason
            
            return MockParseRefusalResponse(f"Invalid JSON response: {str(e)}")

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """Create a regular completion with JSON format."""
        # Reasoning models (gpt-5 family) don't support temperature
        is_reasoning_model = model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3')

        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature if not is_reasoning_model else None,
            max_tokens=max_tokens,
            response_format={'type': 'json_object'},
        )
