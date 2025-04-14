#!/usr/bin/env python3
"""
CLI utility to test an OpenAI-compatible API through the /completions endpoint.
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
        if response.status_code == 200:
            models_data = response.json()
            return [model["id"] for model in models_data.get("data", [])]
        else:
            print(f"Failed to fetch models: {response.status_code} {response.text}")
            return []
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []


def make_completion_request(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 100,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: Optional[List[str]] = None,
    stream: bool = False,
    api_key: str = "sk_test_123",
):
    """Make a request to the completions endpoint."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stream": stream,
    }

    if stop:
        payload["stop"] = stop

    try:
        response = requests.post(
            f"{base_url}/completions", headers=headers, json=payload, stream=stream
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} {response.text}")
            return {"error": response.text}
    except Exception as e:
        print(f"Error making request: {e}")
        return {"error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test OpenAI-compatible completions API"
    )
    parser.add_argument(
        "--endpoint", default="http://localhost:3000/v1", help="API endpoint base URL"
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="Model name (use 'auto' to fetch available models)",
    )
    parser.add_argument(
        "--prompt", help="The prompt to complete", default="Alan Turing is"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature for sampling"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Maximum tokens to generate"
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

    # Auto-detect model if needed
    if args.model == "auto":
        models = fetch_models(args.endpoint)
        if models:
            args.model = models[0]
            print(f"Using model: {args.model}")
        else:
            print("No models found and no model specified. Exiting.")
            sys.exit(1)

    # Make the completion request
    if args.stream:
        for chunk in make_completion_request(
            args.endpoint,
            args.model,
            args.prompt,
            args.temperature,
            args.max_tokens,
            args.top_p,
            args.frequency_penalty,
            args.presence_penalty,
            args.stop,
            args.stream,
            args.api_key,
        ):
            if "choices" in chunk and len(chunk["choices"]) > 0:
                text = chunk["choices"][0].get("text", "")
                print(text, end="")
                sys.stdout.flush()
        print()  # Final newline
    else:
        print(f"Making request to {args.endpoint}/completions")
        result = make_completion_request(
            args.endpoint,
            args.model,
            args.prompt,
            args.temperature,
            args.max_tokens,
            args.top_p,
            args.frequency_penalty,
            args.presence_penalty,
            args.stop,
            False,
            args.api_key,
        )
        # Print the full response in pretty format
        print(json.dumps(result, indent=2))

        # Also print just the generated text for convenience
        if "choices" in result and len(result["choices"]) > 0:
            print("\nGenerated text:")
            print(result["choices"][0].get("text", ""))


if __name__ == "__main__":
    main()
