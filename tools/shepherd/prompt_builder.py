from tools.client.req import LLM

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    system_prompt="You are a helpful assistant.",
)


def is_good_question(question):
    score = llm(
        f"Is this a representative and realistic question that users might ask an online chatbot service? Give a score between 0 to 5. Return the score only without anything in addition.\n{question}."
    )
    try:
        score = int(score)
    except ValueError:
        score = 0
    return score
