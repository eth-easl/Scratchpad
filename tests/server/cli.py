#!/usr/bin/env python3
"""
CLI utility to test an OpenAI-compatible API through the /chat/completions endpoint.
"""

import argparse
import json
import sys
import requests
from typing import List, Optional, Dict, Any, Union


def fetch_models(base_url: str) -> List[str]:
    """Fetch available models from the API."""
    try:
        response = requests.get(f"{base_url}/models")
        response.raise_for_status()  # Raise an exception for bad status codes
        models_data = response.json()
        return [model["id"] for model in models_data.get("data", [])]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding models JSON response.")
        return []


def make_chat_completion_request(
    base_url: str,
    model: str,
    prompt: str,  # User's message content
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: Optional[List[str]] = None,
    stream: bool = False,
    api_key: str = "sk_test_123",
) -> Union[Dict[str, Any], requests.Response, None]:
    """Make a request to the chat completions endpoint."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stream": stream,
        "extra_body": {"separate_reasoning": True},
    }

    if stop:
        payload["stop"] = stop

    try:
        response = requests.post(
            f"{base_url}/chat/completions", headers=headers, json=payload, stream=stream
        )
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        if stream:
            return response  # Return the raw response object for streaming
        else:
            return response.json()  # Return parsed JSON for non-streaming
    except requests.exceptions.HTTPError as e:
        print(
            f"HTTP Error: {e.response.status_code} {e.response.text}", file=sys.stderr
        )
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test OpenAI-compatible chat completions API"
    )
    parser.add_argument(
        "--endpoint", default="http://localhost:8080/v1", help="API endpoint base URL"
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="Model name (use 'auto' to fetch first available model)",
    )
    parser.add_argument(
        "--prompt", help="The user message to send", default="Tell me a joke about AI."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for sampling"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--frequency-penalty", type=float, default=0.0, help="Frequency penalty"
    )
    parser.add_argument(
        "--presence-penalty", type=float, default=0.0, help="Presence penalty"
    )
    parser.add_argument("--stop", nargs="+", help="Stop sequences")
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    parser.add_argument("--api-key", default="sk_test_123", help="API key")

    args = parser.parse_args()

    if args.model == "auto":
        print(f"Fetching available models from {args.endpoint}...")
        models = fetch_models(args.endpoint)
        if models:
            args.model = models[0]
            print(f"Using model: {args.model}")
        else:
            print(
                "No models found or failed to fetch models. Please specify a model manually. Exiting."
            )
            sys.exit(1)

    if args.stream:
        response_stream = make_chat_completion_request(
            args.endpoint,
            args.model,
            args.prompt,
            args.temperature,
            args.max_tokens,
            args.top_p,
            args.frequency_penalty,
            args.presence_penalty,
            args.stop,
            True,  # stream = True
            args.api_key,
        )
        if isinstance(response_stream, requests.Response):
            full_response_text = ""
            print("\nStreaming response:")
            for line in response_stream.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        line_str = line_str[len("data: ") :].strip()
                    if line_str == "[DONE]":
                        break
                    if not line_str:  # Skip empty lines that might occur
                        continue
                    try:
                        chunk = json.loads(line_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                print(content, end="")
                                full_response_text += content
                                sys.stdout.flush()
                    except json.JSONDecodeError:
                        print(
                            f"\nError decoding stream chunk: '{line_str}'",
                            file=sys.stderr,
                        )
                        continue
            print()  # Final newline
            # print(f"\nFull streamed text:\n{full_response_text}") # Optional: print full text at end
        elif response_stream is None:
            print("Failed to get a streaming response.", file=sys.stderr)
            sys.exit(1)

    else:  # Non-streaming
        print(
            f"Making request to {args.endpoint}/chat/completions with model {args.model}..."
        )
        result = make_chat_completion_request(
            args.endpoint,
            args.model,
            args.prompt,
            args.temperature,
            args.max_tokens,
            args.top_p,
            args.frequency_penalty,
            args.presence_penalty,
            args.stop,
            False,  # stream = False
            args.api_key,
        )
        if result and "error" not in result:
            print("\nFull response JSON:")
            print(json.dumps(result, indent=2))

            if (
                "choices" in result
                and len(result["choices"]) > 0
                and "message" in result["choices"][0]
            ):
                message_content = result["choices"][0]["message"].get("content")
                if message_content:
                    print("\nGenerated text:")
                    print(message_content)
                else:
                    print("\nNo content found in the first choice's message.")
            else:
                print("\nCould not extract generated text from response.")
        elif result and "error" in result:
            print(f"API Error: {result.get('error')}", file=sys.stderr)
            sys.exit(1)
        else:
            print(
                "Request failed or returned an unexpected structure.", file=sys.stderr
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
