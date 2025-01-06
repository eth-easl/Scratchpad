from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="test",
)

response = client.embeddings.create(model="", input="test")
print("Embedding response:", response)
