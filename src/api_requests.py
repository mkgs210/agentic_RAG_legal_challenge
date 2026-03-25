import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import Union, List, Dict, Type, Optional, Literal
from openai import OpenAI, RateLimitError as OpenAIRateLimitError
import asyncio
from src.api_request_parallel_processor import process_api_requests_from_file
from openai.lib._parsing import type_to_response_format_param 
import tiktoken
import src.prompts as prompts
import requests
from json_repair import repair_json
from pydantic import BaseModel
from copy import deepcopy
from src.local_models import LocalChatModel, DEFAULT_LOCAL_LLM_MODEL

OPENAI_COMPATIBLE_PROVIDERS = {
    "sambanova": {
        "api_key_env": "SAMBANOVA_API_KEY",
        "base_url_env": "SAMBANOVA_BASE_URL",
        "default_base_url": "https://api.sambanova.ai/v1",
        "model_env": "SAMBANOVA_MODEL",
        "default_model": "Qwen3-32B",
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url_env": "OPENROUTER_BASE_URL",
        "default_base_url": "https://openrouter.ai/api/v1",
        "model_env": "OPENROUTER_MODEL",
        "default_model": "qwen/qwen3-235b-a22b:free",
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url_env": "GROQ_BASE_URL",
        "default_base_url": "https://api.groq.com/openai/v1",
        "model_env": "GROQ_MODEL",
        "default_model": "qwen/qwen3-32b",
    },
    "cerebras": {
        "api_key_env": "CEREBRAS_API_KEY",
        "base_url_env": "CEREBRAS_BASE_URL",
        "default_base_url": "https://api.cerebras.ai/v1",
        "model_env": "CEREBRAS_MODEL",
        "default_model": "qwen-3-32b",
    },
    "mistral": {
        "api_key_env": "MISTRAL_API_KEY",
        "base_url_env": "MISTRAL_BASE_URL",
        "default_base_url": "https://api.mistral.ai/v1",
        "model_env": "MISTRAL_MODEL",
        "default_model": "mistral-small-latest",
    },
}


def _load_dotenvs():
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    load_dotenv(repo_root / "env")
    load_dotenv()



class BaseOpenaiProcessor:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.default_model = 'gpt-4o-2024-08-06'
        # self.default_model = 'gpt-4o-mini-2024-07-18',

    def set_up_llm(self):
        _load_dotenvs()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
            )
        return llm

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None, # For deterministic ouptputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        **kwargs,
        ):
        if model is None:
            model = self.default_model
        use_stream = bool(kwargs.pop("stream", False)) and not is_structured
        uses_max_completion_tokens = str(model).startswith("gpt-5")
        params = {
            "model": model,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content}
            ]
        }
        
        # Some OpenAI families only accept the default temperature.
        if "o3-mini" not in model and not uses_max_completion_tokens:
            params["temperature"] = temperature
        if kwargs.get("max_tokens") is not None:
            if uses_max_completion_tokens:
                params["max_completion_tokens"] = kwargs["max_tokens"]
            else:
                params["max_tokens"] = kwargs["max_tokens"]
        if kwargs.get("request_timeout") is not None:
            params["timeout"] = kwargs["request_timeout"]
            
        if use_stream:
            params["stream"] = True
            params["stream_options"] = {"include_usage": True}
            stream = self.llm.chat.completions.create(**params)
            started = time.perf_counter()
            first_token_at = None
            prev_token_at = None
            token_gaps_ms = []
            pieces = []
            usage = None
            model_name = model
            for chunk in stream:
                if getattr(chunk, "model", None):
                    model_name = chunk.model
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta.content if chunk.choices[0].delta else None
                if not delta:
                    continue
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                elif prev_token_at is not None:
                    token_gaps_ms.append((now - prev_token_at) * 1000)
                prev_token_at = now
                pieces.append(delta)

            total_time_ms = int((time.perf_counter() - started) * 1000)
            ttft_ms = int((first_token_at - started) * 1000) if first_token_at is not None else total_time_ms
            tpot_ms = int(sum(token_gaps_ms) / len(token_gaps_ms)) if token_gaps_ms else 0
            content = "".join(pieces)
            self.response_data = {
                "model": model_name,
                "input_tokens": getattr(usage, "prompt_tokens", 0) if usage is not None else 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) if usage is not None else 0,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "total_time_ms": total_time_ms,
                "streaming": True,
            }
            print(self.response_data)

        elif not is_structured:
            completion = self.llm.chat.completions.create(**params)
            content = completion.choices[0].message.content

        elif is_structured:
            params["response_format"] = response_format
            completion = self.llm.beta.chat.completions.parse(**params)

            response = completion.choices[0].message.parsed
            content = response.dict()

        if not use_stream:
            self.response_data = {"model": completion.model, "input_tokens": completion.usage.prompt_tokens, "output_tokens": completion.usage.completion_tokens}
            print(self.response_data)

        return content

    @staticmethod
    def count_tokens(string, encoding_name="o200k_base"):
        encoding = tiktoken.get_encoding(encoding_name)

        # Encode the string and count the tokens
        tokens = encoding.encode(string)
        token_count = len(tokens)

        return token_count


class BaseIBMAPIProcessor:
    def __init__(self):
        _load_dotenvs()
        self.api_token = os.getenv("IBM_API_KEY")
        self.base_url = "https://rag.timetoact.at/ibm"
        self.default_model = 'meta-llama/llama-3-3-70b-instruct'
    def check_balance(self):
        """Check the current balance for the provided token."""
        balance_url = f"{self.base_url}/balance"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            response = requests.get(balance_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error checking balance: {err}")
            return None
    
    def get_available_models(self):
        """Get a list of available foundation models."""
        models_url = f"{self.base_url}/foundation_model_specs"
        
        try:
            response = requests.get(models_url)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting available models: {err}")
            return None
    
    def get_embeddings(self, texts, model_id="ibm/granite-embedding-278m-multilingual"):
        """Get vector embeddings for the provided text inputs."""
        embeddings_url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": texts,
            "model_id": model_id
        }
        
        try:
            response = requests.post(embeddings_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting embeddings: {err}")
            return None
    
    def send_message(
        self,
        # model='meta-llama/llama-3-1-8b-instruct',
        model=None,
        temperature=0.5,
        seed=None,  # For deterministic outputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        max_new_tokens=5000,
        min_new_tokens=1,
        **kwargs
    ):
        if model is None:
            model = self.default_model
        text_generation_url = f"{self.base_url}/text_generation"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Prepare the input messages
        input_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content}
        ]
        
        # Prepare parameters with defaults and any additional parameters
        parameters = {
            "temperature": temperature,
            "random_seed": seed,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            **kwargs
        }
        
        payload = {
            "input": input_messages,
            "model_id": model,
            "parameters": parameters
        }
        
        try:
            response = requests.post(text_generation_url, headers=headers, json=payload)
            response.raise_for_status()
            completion = response.json()

            content = completion.get("results")[0].get("generated_text")
            self.response_data = {"model": completion.get("model_id"), "input_tokens": completion.get("results")[0].get("input_token_count"), "output_tokens": completion.get("results")[0].get("generated_token_count")}
            print(self.response_data)
            if is_structured and response_format is not None:
                try:
                    repaired_json = repair_json(content)
                    parsed_dict = json.loads(repaired_json)
                    validated_data = response_format.model_validate(parsed_dict)
                    content = validated_data.model_dump()
                    return content
                
                except Exception as err:
                    print("Error processing structured response, attempting to reparse the response...")
                    reparsed = self._reparse_response(content, system_content)
                    try:
                        repaired_json = repair_json(reparsed)
                        reparsed_dict = json.loads(repaired_json)
                        try:
                            validated_data = response_format.model_validate(reparsed_dict)
                            print("Reparsing successful!")
                            content = validated_data.model_dump()
                            return content
                        
                        except Exception:
                            return reparsed_dict
                        
                    except Exception as reparse_err:
                        print(f"Reparse failed with error: {reparse_err}")
                        print(f"Reparsed response: {reparsed}")
                        return content
            
            return content

        except requests.HTTPError as err:
            print(f"Error generating text: {err}")
            return None

    def _reparse_response(self, response, system_content):

        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=system_content,
            response=response
        )
        
        reparsed_response = self.send_message(
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )
        
        return reparsed_response

     
class BaseGeminiProcessor:
    def __init__(self):
        _load_dotenvs()
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        self.default_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.thinking_budget = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
        self.max_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "5"))
        self.min_interval_seconds = float(os.getenv("GEMINI_MIN_INTERVAL_SECONDS", "0"))
        self._last_request_monotonic = 0.0
        if not self.api_key:
            raise ValueError("Missing API key for provider 'gemini' in GEMINI_API_KEY")
        
    def list_available_models(self) -> None:
        """
        Prints available Gemini models that support text generation.
        """
        response = requests.get(
            f"{self.base_url}/models",
            headers={"x-goog-api-key": self.api_key},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        print("Available Gemini models for text generation:")
        for model in payload.get("models", []):
            methods = model.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                continue
            print(f"- {model.get('name')}")
            print(f"  Input token limit: {model.get('inputTokenLimit')}")
            print(f"  Output token limit: {model.get('outputTokenLimit')}")
            print()

    @staticmethod
    def _parse_retry_seconds(response: requests.Response) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except Exception:
                pass
        try:
            payload = response.json()
        except Exception:
            return 20.0
        for detail in payload.get("error", {}).get("details", []):
            delay = detail.get("retryDelay")
            if not delay:
                continue
            if isinstance(delay, str) and delay.endswith("s"):
                try:
                    return max(float(delay[:-1]), 1.0)
                except Exception:
                    continue
        return 20.0

    def _respect_min_interval(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_monotonic
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

    def _generate_with_retry(self, url, payload, timeout):
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._respect_min_interval()
            response = requests.post(
                url,
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            self._last_request_monotonic = time.monotonic()
            if response.status_code < 400:
                return response.json()

            error = requests.HTTPError(response.text, response=response)
            last_error = error
            if response.status_code != 429 or attempt == self.max_attempts:
                raise error

            wait_seconds = self._parse_retry_seconds(response)
            print(f"\nGemini rate limited on attempt {attempt}/{self.max_attempts}. Waiting {wait_seconds:.1f}s before retry...\n")
            time.sleep(wait_seconds)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini request failed without response")

    def _parse_structured_response(self, response_text, response_format, model):
        try:
            repaired_json = repair_json(response_text)
            parsed_dict = json.loads(repaired_json)
            parsed_dict = self._coerce_structured_payload(parsed_dict, response_format)
            validated_data = response_format.model_validate(parsed_dict)
            return validated_data.model_dump()
        except Exception as err:
            print(f"Error parsing structured response: {err}")
            print("Attempting to reparse the response...")
            reparsed = self._reparse_response(response_text, response_format, model)
            return reparsed

    @staticmethod
    def _coerce_structured_payload(payload, response_format):
        if not isinstance(payload, dict):
            return payload
        coerced = dict(payload)
        for field_name, field_info in response_format.model_fields.items():
            value = coerced.get(field_name)
            annotation = field_info.annotation
            origin = getattr(annotation, "__origin__", None)
            if origin is list and isinstance(value, str):
                coerced[field_name] = [value]
            elif origin is list and value is None:
                coerced[field_name] = []
            elif annotation is str and value is None:
                coerced[field_name] = ""
        return coerced

    def _reparse_response(self, response, response_format, model):
        """Reparse invalid JSON responses using the model itself."""
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=prompts.AnswerSchemaFixPrompt.system_prompt,
            response=response
        )
        
        try:
            reparsed_response = self.send_message(
                model=model,
                system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
                human_content=user_prompt,
                is_structured=False
            )
            
            try:
                repaired_json = repair_json(reparsed_response)
                reparsed_dict = json.loads(repaired_json)
                reparsed_dict = self._coerce_structured_payload(reparsed_dict, response_format)
                try:
                    validated_data = response_format.model_validate(reparsed_dict)
                    print("Reparsing successful!")
                    return validated_data.model_dump()
                except Exception:
                    return reparsed_dict
            except Exception as reparse_err:
                print(f"Reparse failed with error: {reparse_err}")
                print(f"Reparsed response: {reparsed_response}")
                return response
        except Exception as e:
            print(f"Reparse attempt failed: {e}")
            return response

    def send_message(
        self,
        model=None,
        temperature: float = 0.5,
        seed=12345,  # For back compatibility
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        is_structured: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> Union[str, Dict, None]:
        if model is None:
            model = self.default_model

        generation_config = {"temperature": temperature}
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if self.thinking_budget >= 0:
            generation_config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}

        prompt = f"{system_content}\n\n---\n\n{human_content}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": generation_config,
        }
        request_timeout = kwargs.get("request_timeout", 120)
        url = f"{self.base_url}/models/{model}:generateContent"

        try:
            response = self._generate_with_retry(url, payload, request_timeout)
            candidates = response.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            text_parts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
            content = "\n".join(text_parts).strip()
            usage = response.get("usageMetadata", {})

            self.response_data = {
                "model": response.get("modelVersion", model),
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0),
            }
            print(self.response_data)
            
            if is_structured and response_format is not None:
                return self._parse_structured_response(content, response_format, model)
            
            return content
        except Exception as e:
            raise Exception(f"API request failed after retries: {str(e)}")


class BaseLocalProcessor:
    def __init__(self):
        self.default_model = DEFAULT_LOCAL_LLM_MODEL
        self.max_new_tokens = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "1536"))

    def _parse_structured_response(self, response_text, response_format, model, original_system_prompt):
        try:
            repaired_json = repair_json(response_text)
            parsed_dict = json.loads(repaired_json)
            validated_data = response_format.model_validate(parsed_dict)
            return validated_data.model_dump()
        except Exception as err:
            print(f"Error parsing structured response: {err}")
            print("Attempting to reparse the response...")
            return self._reparse_response(response_text, response_format, model, original_system_prompt)

    def _reparse_response(self, response, response_format, model, original_system_prompt):
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=original_system_prompt,
            response=response
        )

        reparsed_response = self.send_message(
            model=model,
            temperature=0,
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )

        try:
            repaired_json = repair_json(reparsed_response)
            reparsed_dict = json.loads(repaired_json)
            validated_data = response_format.model_validate(reparsed_dict)
            print("Reparsing successful!")
            return validated_data.model_dump()
        except Exception as err:
            raise ValueError(f"Failed to parse structured local response: {err}") from err

    def send_message(
        self,
        model=None,
        temperature=0.0,
        seed=None,
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        if model is None:
            model = self.default_model

        llm = LocalChatModel(model_name=model)
        max_new_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_new_tokens))
        content, usage = llm.generate(
            system_content=system_content,
            human_content=human_content,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        self.response_data = usage
        print(self.response_data)

        if is_structured and response_format is not None:
            return self._parse_structured_response(content, response_format, model, system_content)

        return content


class BaseOpenAICompatibleProcessor:
    def __init__(self, provider_name: str):
        if provider_name not in OPENAI_COMPATIBLE_PROVIDERS:
            raise ValueError(f"Unsupported OpenAI-compatible provider: {provider_name}")

        _load_dotenvs()
        config = OPENAI_COMPATIBLE_PROVIDERS[provider_name]
        self.provider_name = provider_name
        self.api_key = os.getenv(config["api_key_env"])
        self.base_url = os.getenv(config["base_url_env"], config["default_base_url"])
        self.default_model = os.getenv(config["model_env"], config["default_model"])

        if not self.api_key:
            raise ValueError(f"Missing API key for provider '{provider_name}' in {config['api_key_env']}")

        client_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": None,
            "max_retries": 2,
        }

        if self.provider_name == "openrouter":
            headers = {}
            if os.getenv("OPENROUTER_HTTP_REFERER"):
                headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER")
            if os.getenv("OPENROUTER_APP_NAME", "hackaton"):
                headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "hackaton")
            if headers:
                client_kwargs["default_headers"] = headers

        self.llm = OpenAI(**client_kwargs)

    def _create_completion(self, params):
        wait_seconds = int(
            os.getenv(
                f"{self.provider_name.upper()}_RATE_LIMIT_WAIT",
                "35" if self.provider_name == "mistral" else "10",
            )
        )
        max_attempts = int(os.getenv(f"{self.provider_name.upper()}_MAX_ATTEMPTS", "4"))

        for attempt in range(1, max_attempts + 1):
            try:
                return self.llm.chat.completions.create(**params)
            except OpenAIRateLimitError:
                if attempt == max_attempts:
                    raise
                print(f"Rate limited by {self.provider_name}; retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)

    @staticmethod
    def _extract_message_content(message_content):
        if isinstance(message_content, str):
            return message_content
        if isinstance(message_content, list):
            parts = []
            for part in message_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif hasattr(part, "text"):
                    parts.append(part.text)
                else:
                    parts.append(str(part))
            return "\n".join(part for part in parts if part).strip()
        return str(message_content)

    @staticmethod
    def _coerce_structured_payload(payload):
        if isinstance(payload, list):
            dict_items = [item for item in payload if isinstance(item, dict)]
            if dict_items:
                return dict_items[-1]
            nested_lists = [item for item in payload if isinstance(item, list)]
            for item in nested_lists:
                coerced = BaseOpenAICompatibleProcessor._coerce_structured_payload(item)
                if isinstance(coerced, dict):
                    return coerced
            string_items = [item for item in payload if isinstance(item, str)]
            if string_items:
                try:
                    reparsed = json.loads(repair_json("\n".join(string_items)))
                    return BaseOpenAICompatibleProcessor._coerce_structured_payload(reparsed)
                except Exception:
                    return payload
        return payload

    def _parse_structured_response(self, response_text, response_format, model, original_system_prompt):
        try:
            repaired_json = repair_json(response_text)
            parsed_dict = json.loads(repaired_json)
            parsed_dict = self._coerce_structured_payload(parsed_dict)
            validated_data = response_format.model_validate(parsed_dict)
            return validated_data.model_dump()
        except Exception as err:
            print(f"Error parsing structured response for {self.provider_name}: {err}")
            print("Attempting to reparse the response...")
            return self._reparse_response(response_text, response_format, model, original_system_prompt)

    def _reparse_response(self, response, response_format, model, original_system_prompt):
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=original_system_prompt,
            response=response
        )

        reparsed_response = self.send_message(
            model=model,
            temperature=0,
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )

        repaired_json = repair_json(reparsed_response)
        reparsed_dict = json.loads(repaired_json)
        reparsed_dict = self._coerce_structured_payload(reparsed_dict)
        validated_data = response_format.model_validate(reparsed_dict)
        print("Reparsing successful!")
        return validated_data.model_dump()

    def send_message(
        self,
        model=None,
        temperature=0.2,
        seed=None,
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        if model is None:
            model = self.default_model

        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content},
            ],
        }

        if temperature is not None:
            params["temperature"] = temperature
        params.update({key: value for key, value in kwargs.items() if value is not None})

        completion = self._create_completion(params)
        content = self._extract_message_content(completion.choices[0].message.content)
        self.response_data = {
            "model": completion.model,
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        }
        print(self.response_data)

        if is_structured and response_format is not None:
            return self._parse_structured_response(content, response_format, model, system_content)

        return content


class BaseCohereProcessor:
    def __init__(self):
        _load_dotenvs()
        self.api_key = os.getenv("COHERE_API_KEY")
        self.base_url = os.getenv("COHERE_BASE_URL", "https://api.cohere.com/v2")
        self.default_model = os.getenv("COHERE_MODEL", "command-a-03-2025")

        if not self.api_key:
            raise ValueError("Missing API key for provider 'cohere' in COHERE_API_KEY")

    @staticmethod
    def _extract_message_content(response_json):
        content_items = response_json.get("message", {}).get("content", [])
        if isinstance(content_items, str):
            return content_items

        parts = []
        for item in content_items:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    def _parse_structured_response(self, response_text, response_format, model, original_system_prompt):
        try:
            repaired_json = repair_json(response_text)
            parsed_dict = json.loads(repaired_json)
            validated_data = response_format.model_validate(parsed_dict)
            return validated_data.model_dump()
        except Exception as err:
            print(f"Error parsing structured response for cohere: {err}")
            print("Attempting to reparse the response...")
            return self._reparse_response(response_text, response_format, model, original_system_prompt)

    def _reparse_response(self, response, response_format, model, original_system_prompt):
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=original_system_prompt,
            response=response
        )

        reparsed_response = self.send_message(
            model=model,
            temperature=0,
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )

        repaired_json = repair_json(reparsed_response)
        reparsed_dict = json.loads(repaired_json)
        validated_data = response_format.model_validate(reparsed_dict)
        print("Reparsing successful!")
        return validated_data.model_dump()

    def send_message(
        self,
        model=None,
        temperature=0.2,
        seed=None,
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        if model is None:
            model = self.default_model

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content},
            ],
            "temperature": temperature,
        }
        payload.update({key: value for key, value in kwargs.items() if value is not None})
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        wait_seconds = int(os.getenv("COHERE_RATE_LIMIT_WAIT", "10"))
        max_attempts = int(os.getenv("COHERE_MAX_ATTEMPTS", "4"))

        response = None
        for attempt in range(1, max_attempts + 1):
            response = requests.post(
                f"{self.base_url}/chat",
                headers=headers,
                json=payload,
                timeout=120,
            )
            if response.status_code != 429 or attempt == max_attempts:
                break
            print(f"Rate limited by cohere; retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)

        response.raise_for_status()
        completion = response.json()
        content = self._extract_message_content(completion)

        usage = completion.get("usage", {}).get("tokens", {})
        self.response_data = {
            "model": model,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
        }
        print(self.response_data)

        if is_structured and response_format is not None:
            return self._parse_structured_response(content, response_format, model, system_content)

        return content


class BaseAnthropicProcessor:
    def __init__(self):
        _load_dotenvs()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        self.default_model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.max_attempts = int(os.getenv("ANTHROPIC_MAX_ATTEMPTS", "5"))
        self.min_interval_seconds = float(os.getenv("ANTHROPIC_MIN_INTERVAL_SECONDS", "0"))
        self.thinking_budget = int(os.getenv("ANTHROPIC_THINKING_BUDGET", "0"))
        self._last_request_monotonic = 0.0

        if not self.api_key:
            raise ValueError("Missing API key for provider 'anthropic' in ANTHROPIC_API_KEY")

    @staticmethod
    def _extract_message_content(response_json):
        content_items = response_json.get("content", [])
        if isinstance(content_items, str):
            return content_items
        parts = []
        for item in content_items:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part).strip()

    @staticmethod
    def _parse_retry_seconds(response: requests.Response) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except Exception:
                pass
        return 20.0

    def _respect_min_interval(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_monotonic
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

    def _parse_structured_response(self, response_text, response_format, model, original_system_prompt):
        try:
            repaired_json = repair_json(response_text)
            parsed_dict = json.loads(repaired_json)
            parsed_dict = self._coerce_structured_payload(parsed_dict, response_format)
            validated_data = response_format.model_validate(parsed_dict)
            return validated_data.model_dump()
        except Exception as err:
            print(f"Error parsing structured response for anthropic: {err}")
            print("Attempting to reparse the response...")
            return self._reparse_response(response_text, response_format, model, original_system_prompt)

    @staticmethod
    def _coerce_structured_payload(payload, response_format):
        if not isinstance(payload, dict):
            return payload
        coerced = dict(payload)
        for field_name, field_info in response_format.model_fields.items():
            value = coerced.get(field_name)
            annotation = field_info.annotation
            origin = getattr(annotation, "__origin__", None)
            if origin is list and isinstance(value, str):
                coerced[field_name] = [value]
            elif origin is list and value is None:
                coerced[field_name] = []
            elif annotation is str and value is None:
                coerced[field_name] = ""
        return coerced

    def _reparse_response(self, response, response_format, model, original_system_prompt):
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=original_system_prompt,
            response=response
        )

        reparsed_response = self.send_message(
            model=model,
            temperature=0,
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )

        repaired_json = repair_json(reparsed_response)
        reparsed_dict = json.loads(repaired_json)
        reparsed_dict = self._coerce_structured_payload(reparsed_dict, response_format)
        validated_data = response_format.model_validate(reparsed_dict)
        print("Reparsing successful!")
        return validated_data.model_dump()

    def send_message(
        self,
        model=None,
        temperature=0.2,
        seed=None,
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        if model is None:
            model = self.default_model

        payload = {
            "model": model,
            "system": system_content,
            "messages": [
                {"role": "user", "content": human_content},
            ],
            "temperature": temperature,
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        if self.thinking_budget > 0:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        timeout = kwargs.get("request_timeout", 120)
        response = None
        for attempt in range(1, self.max_attempts + 1):
            self._respect_min_interval()
            response = requests.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            self._last_request_monotonic = time.monotonic()
            if response.status_code < 400:
                break
            if response.status_code != 429 or attempt == self.max_attempts:
                raise requests.HTTPError(response.text, response=response)
            wait_seconds = self._parse_retry_seconds(response)
            print(f"Anthropic rate limited on attempt {attempt}/{self.max_attempts}; waiting {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)

        if response is None:
            raise RuntimeError("Anthropic request failed without response")

        completion = response.json()
        content = self._extract_message_content(completion)
        usage = completion.get("usage", {})
        self.response_data = {
            "model": completion.get("model", model),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
        print(self.response_data)

        if is_structured and response_format is not None:
            return self._parse_structured_response(content, response_format, model, system_content)

        return content


class APIProcessor:
    def __init__(self, provider: Literal["openai", "ibm", "gemini", "anthropic", "local", "sambanova", "openrouter", "groq", "cerebras", "mistral", "cohere"] ="openai"):
        self.provider = provider.lower()
        if self.provider == "openai":
            self.processor = BaseOpenaiProcessor()
        elif self.provider == "ibm":
            self.processor = BaseIBMAPIProcessor()
        elif self.provider == "gemini":
            self.processor = BaseGeminiProcessor()
        elif self.provider == "anthropic":
            self.processor = BaseAnthropicProcessor()
        elif self.provider == "local":
            self.processor = BaseLocalProcessor()
        elif self.provider == "cohere":
            self.processor = BaseCohereProcessor()
        elif self.provider in OPENAI_COMPATIBLE_PROVIDERS:
            self.processor = BaseOpenAICompatibleProcessor(self.provider)
        else:
            raise ValueError(f"Unsupported API provider: {provider}")

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """
        Routes the send_message call to the appropriate processor.
        The underlying processor's send_message method is responsible for handling the parameters.
        """
        if model is None:
            model = self.processor.default_model
        return self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            **kwargs
        )

    def get_answer_from_rag_context(self, question, rag_context, schema, model):
        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(schema)
        
        answer_dict = self.processor.send_message(
            model=model,
            system_content=system_prompt,
            human_content=user_prompt.format(context=rag_context, question=question),
            is_structured=True,
            response_format=response_format
        )
        self.response_data = self.processor.response_data
        return answer_dict


    def _build_rag_context_prompts(self, schema):
        """Return prompts tuple for the given schema."""
        use_schema_prompt = self.provider in {"ibm", "gemini", "anthropic", "local", "sambanova", "openrouter", "groq", "cerebras", "mistral", "cohere"}
        
        if schema == "name":
            system_prompt = (prompts.AnswerWithRAGContextNamePrompt.system_prompt_with_schema 
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamePrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamePrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamePrompt.user_prompt
        elif schema == "number":
            system_prompt = (prompts.AnswerWithRAGContextNumberPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNumberPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNumberPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNumberPrompt.user_prompt
        elif schema == "boolean":
            system_prompt = (prompts.AnswerWithRAGContextBooleanPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextBooleanPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextBooleanPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextBooleanPrompt.user_prompt
        elif schema == "names":
            system_prompt = (prompts.AnswerWithRAGContextNamesPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamesPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamesPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamesPrompt.user_prompt
        elif schema == "comparative":
            system_prompt = (prompts.ComparativeAnswerPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.ComparativeAnswerPrompt.system_prompt)
            response_format = prompts.ComparativeAnswerPrompt.AnswerSchema
            user_prompt = prompts.ComparativeAnswerPrompt.user_prompt
        else:
            raise ValueError(f"Unsupported schema: {schema}")
        return system_prompt, response_format, user_prompt

    def get_rephrased_questions(self, original_question: str, companies: List[str]) -> Dict[str, str]:
        """Use LLM to break down a comparative question into individual questions."""
        answer_dict = self.processor.send_message(
            system_content=prompts.RephrasedQuestionsPrompt.system_prompt,
            human_content=prompts.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question,
                companies=", ".join([f'"{company}"' for company in companies])
            ),
            is_structured=True,
            response_format=prompts.RephrasedQuestionsPrompt.RephrasedQuestions
        )
        
        # Convert the answer_dict to the desired format
        questions_dict = {item["company_name"]: item["question"] for item in answer_dict["questions"]}
        
        return questions_dict


class AsyncOpenaiProcessor:
    
    def _get_unique_filepath(self, base_filepath):
        """Helper method to get unique filepath"""
        if not os.path.exists(base_filepath):
            return base_filepath
        
        base, ext = os.path.splitext(base_filepath)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        return f"{base}_{counter}{ext}"

    async def process_structured_ouputs_requests(
        self,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        queries=None,
        response_format=None,
        requests_filepath='./temp_async_llm_requests.jsonl',
        save_filepath='./temp_async_llm_results.jsonl',
        preserve_requests=False,
        preserve_results=True,
        request_url="https://api.openai.com/v1/chat/completions",
        max_requests_per_minute=3_500,
        max_tokens_per_minute=3_500_000,
        token_encoding_name="o200k_base",
        max_attempts=5,
        logging_level=20,
        progress_callback=None
    ):
        # Create requests for jsonl
        jsonl_requests = []
        for idx, query in enumerate(queries):
            request = {
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query},
                ],
                'response_format': type_to_response_format_param(response_format),
                'metadata': {'original_index': idx}
            }
            jsonl_requests.append(request)
            
        # Get unique filepaths if files already exist
        requests_filepath = self._get_unique_filepath(requests_filepath)
        save_filepath = self._get_unique_filepath(save_filepath)

        # Write requests to JSONL file
        with open(requests_filepath, "w") as f:
            for request in jsonl_requests:
                json_string = json.dumps(request)
                f.write(json_string + "\n")

        # Process API requests
        total_requests = len(jsonl_requests)

        async def monitor_progress():
            last_count = 0
            while True:
                try:
                    with open(save_filepath, 'r') as f:
                        current_count = sum(1 for _ in f)
                        if current_count > last_count:
                            if progress_callback:
                                for _ in range(current_count - last_count):
                                    progress_callback()
                            last_count = current_count
                        if current_count >= total_requests:
                            break
                except FileNotFoundError:
                    pass
                await asyncio.sleep(0.1)

        async def process_with_progress():
            await asyncio.gather(
                process_api_requests_from_file(
                    requests_filepath=requests_filepath,
                    save_filepath=save_filepath,
                    request_url=request_url,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    max_requests_per_minute=max_requests_per_minute,
                    max_tokens_per_minute=max_tokens_per_minute,
                    token_encoding_name=token_encoding_name,
                    max_attempts=max_attempts,
                    logging_level=logging_level
                ),
                monitor_progress()
            )

        await process_with_progress()

        with open(save_filepath, "r") as f:
            validated_data_list = []
            results = []
            for line_number, line in enumerate(f, start=1):
                raw_line = line.strip()
                try:
                    result = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Line {line_number}: Failed to load JSON from line: {raw_line}")
                    continue

                # Check finish_reason in the API response
                finish_reason = result[1]['choices'][0].get('finish_reason', '')
                if finish_reason != "stop":
                    print(f"[WARNING] Line {line_number}: finish_reason is '{finish_reason}' (expected 'stop').")

                # Safely parse answer; if it fails, leave answer empty and report the error.
                try:
                    answer_content = result[1]['choices'][0]['message']['content']
                    answer_parsed = json.loads(answer_content)
                    answer = response_format(**answer_parsed).model_dump()
                except Exception as e:
                    print(f"[ERROR] Line {line_number}: Failed to parse answer JSON. Error: {e}.")
                    answer = ""

                results.append({
                    'index': result[2],
                    'question': result[0]['messages'],
                    'answer': answer
                })
            
            # Sort by original index and build final list
            validated_data_list = [
                {'question': r['question'], 'answer': r['answer']} 
                for r in sorted(results, key=lambda x: x['index']['original_index'])
            ]

        if not preserve_requests:
            os.remove(requests_filepath)

        if not preserve_results:
            os.remove(save_filepath)
        else:  # Fix requests order
            with open(save_filepath, "r") as f:
                results = [json.loads(line) for line in f]
            
            sorted_results = sorted(results, key=lambda x: x[2]['original_index'])
            
            with open(save_filepath, "w") as f:
                for result in sorted_results:
                    json_string = json.dumps(result)
                    f.write(json_string + "\n")
            
        return validated_data_list
