from tools.client.req import LLM
import multiprocessing as mp

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    system_prompt="You are a helpful assistant.",
)
available_models = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
]


def is_good_question(question):
    score = llm(
        f"Is this a representative and realistic question that users might ask an online chatbot service? Give a score between 0 to 5. Return the score only without anything in addition.\n{question}."
    )
    try:
        score = int(score)
    except ValueError:
        score = 0
    return score


def get_response_models(models, questions):
    responses = []
    for model in models:
        llm = LLM(
            model=model,
            system_prompt="You are a helpful assistant.",
        )
        with mp.Pool(4) as pool:
            response = pool.map(llm, questions)
        for i, r in enumerate(response):
            responses.append({"model": model, "question": questions[i], "response": r})
    return responses
