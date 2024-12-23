import multiprocessing as mp
from scratchpad.utils.client import LLM

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


def judge_response(judge_model, question, response):
    prompt_template = """
    You will be given a user_question and system_answer couple.
    Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
    Give your answer on a scale of 1 to 10, where 1 means that the system_answer is not helpful at all, and 10 means that the system_answer completely and helpfully addresses the user_question.

    Here is the scale you should use to build your answer:
    1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
    3: The system_answer is mostly not helpful: misses some key aspects of the question
    6: The system_answer is mostly helpful: provides support, but still could be improved
    10: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

    Provide your feedback as follows:

    Feedback:::
    Evaluation: (your rationale for the rating, as a text)
    Total rating: (your rating, as a number between 1 and 10)

    You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

    Now here are the question and answer.

    Question: {question}
    Answer: {answer}

    Provide your feedback.
    Feedback:::
    Evaluation: """
    llm = LLM(
        model=judge_model,
        system_prompt="You are a helpful assistant. You are evaluating the quality of a chatbot response and you are supposed to be critical, honest and reflect the response quality of the chatbot.",
    )
    score = llm(prompt_template.format(question=question, answer=response))
    try:
        final_score = int(score.split("Total rating: ")[1].split("\n")[0])
        reason = score.split("Evaluation: ")[1].split("\n")[0]
    except:
        final_score = 0
        reason = "Error"
    return final_score, reason
