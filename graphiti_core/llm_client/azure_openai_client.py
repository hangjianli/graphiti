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
import logging
from typing import Any, ClassVar

from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .config import DEFAULT_MAX_TOKENS, LLMConfig
from .openai_base_client import BaseOpenAIClient

logger = logging.getLogger(__name__)


class AzureOpenAILLMClient(BaseOpenAIClient):
    """Wrapper class for AsyncAzureOpenAI that implements the LLMClient interface."""

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        azure_client: AsyncAzureOpenAI,
        config: LLMConfig | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        super().__init__(
            config,
            cache=False,
            max_tokens=max_tokens,
            reasoning=reasoning,
            verbosity=verbosity,
        )
        self.client = azure_client

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None,
        verbosity: str | None,
    ):
        """Create a structured completion using Azure OpenAI's responses.parse API."""
        supports_reasoning = self._supports_reasoning_features(model)
        request_kwargs = {
            'model': model,
            'input': messages,
            'max_output_tokens': max_tokens,
            'text_format': response_model,  # type: ignore
        }

        temperature_value = temperature if not supports_reasoning else None
        if temperature_value is not None:
            request_kwargs['temperature'] = temperature_value

        if supports_reasoning and reasoning:
            request_kwargs['reasoning'] = {'effort': reasoning}  # type: ignore

        if supports_reasoning and verbosity:
            request_kwargs['text'] = {'verbosity': verbosity}  # type: ignore
        
        # Use custom parse implementation instead of responses.parse
        return await self._custom_parse(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=response_model,
            reasoning=reasoning,
            verbosity=verbosity,
        )

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
    ):
        """Create a regular completion with JSON format using Azure OpenAI."""
        supports_reasoning = self._supports_reasoning_features(model)

        request_kwargs = {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens,
            'response_format': {'type': 'json_object'},
        }

        temperature_value = temperature if not supports_reasoning else None
        if temperature_value is not None:
            request_kwargs['temperature'] = temperature_value

        return await self.client.chat.completions.create(**request_kwargs)

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
        supports_reasoning = self._supports_reasoning_features(model)
        
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
        
        # Get the schema and ensure compliance with OpenAI strict mode requirements
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
        
        # Create completion with structured output
        request_kwargs = {
            'model': model,
            'messages': enhanced_messages,
            'max_tokens': max_tokens,
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'name': response_model.__name__,
                    'schema': schema,
                    'strict': True
                }
            }
        }
        
        temperature_value = temperature if not supports_reasoning else None
        if temperature_value is not None:
            request_kwargs['temperature'] = temperature_value
        
        response = await self.client.chat.completions.create(**request_kwargs)
        
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

    @staticmethod
    def _supports_reasoning_features(model: str) -> bool:
        """Return True when the Azure model supports reasoning/verbosity options."""
        reasoning_prefixes = ('o1', 'o3', 'gpt-5')
        return model.startswith(reasoning_prefixes)
