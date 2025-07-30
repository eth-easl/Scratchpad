import requests
import json


def test_openai_compatible_api():
    """Test OpenAI-compatible API with a simple chat completion request."""

    # API configuration
    base_url = "http://localhost:8080/v1"  # Adjust to your API endpoint
    api_key = "your-api-key"  # Replace with actual API key if needed

    # Headers
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    # Test data
    payload = {
        "model": "Qwen/Qwen3-8B",  # Adjust model name as needed
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Can you help me test this API?"},
        ],
        "max_tokens": 2048,
        "temperature": 0.7,
        "return_raw": True,  # Set to True to return raw prompts
    }

    try:
        # Make the request
        response = requests.post(
            f"{base_url}/chat/completions", headers=headers, json=payload, timeout=30
        )

        # Check response status
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ API request successful!")
            print(f"Response: {json.dumps(result, indent=2)}")

            # Validate response structure
            assert "choices" in result
            assert len(result["choices"]) > 0
            assert "message" in result["choices"][0]
            assert "content" in result["choices"][0]["message"]

            print("✅ Response structure validation passed!")

        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
    except AssertionError as e:
        print(f"❌ Response validation failed: {e}")


if __name__ == "__main__":
    print("🚀 Testing OpenAI-compatible API...")
    test_openai_compatible_api()
    print("\n✨ Test completed!")
