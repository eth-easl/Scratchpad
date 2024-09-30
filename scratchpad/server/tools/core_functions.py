import requests
import openai


def calculator(a, b, operator):
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return a / b
    else:
        return "Invalid operator"


def generate(prompt):
    client = openai.OpenAI(
        base_url="http://localhost:3000/v1",
        api_key="sk_test_123",
    )
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
        max_tokens=50,
        top_p=0.9,
    )
    return response


def wiki(topic):
    related_topics = generate(
        f"Generate four related topic about {topic} that might be an article in Wikipedia. Just use two words separated by a comma without anything else."
    )
    topics = related_topics.choices[0].message.content.split(",")
    endpoint = "https://en.wikipedia.org/w/api.php"
    topics = [topic.strip() for topic in topics] + [topic]
    extracts = []
    for topic in topics:
        try:
            params = {
                "action": "query",
                "format": "json",
                "titles": topic,
                "prop": "extracts",
                "explaintext": True,
                "exsectionformat": "wiki",
                "redirects": True,
                "exsentences": 3,  # get first 3 sentences of the page
            }
            response = requests.get(endpoint, params=params).json()
            page_id = list(response["query"]["pages"].keys())[0]
            page_extract = response["query"]["pages"][page_id]["extract"]
            extracts.append(page_extract)
        except Exception as e:
            extracts.append(f"I couldn't find any information about {topic}.")
    extracts = "\n\n".join(extracts)
    return extracts


def search(topic):
    related_topics = generate(
        f"Generate four related topic about {topic} that might be an article in Wikipedia. Just use two words separated by a comma without anything else."
    )
    topics = related_topics.choices[0].message.content.split(",")
    # using duckduckgo
    topics = [topic.strip() for topics in topics] + [topic]
    search_results = []
    topics = "%2d".join(topics)
    endpoint = f"http://api.duckduckgo.com/?q={topic}&format=json&pretty=1&no_html=1&skip_disambig=1"
    results = requests.get(endpoint).json()
    search_results.append(results["AbstractText"])
    search_results = "\n\n".join(search_results)
    return search_results
