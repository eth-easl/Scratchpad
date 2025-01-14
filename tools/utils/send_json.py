import json
import requests


def send_json(args):
    json_obj = json.loads(args.json)
    url = f"{args.url}/{args.endpoint}"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=json_obj)
    if response.status_code == 200:
        print("JSON Object sent successfully")
    else:
        print("Failed to send JSON Object")
    print(response.json())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send JSON Object to local server")
    parser.add_argument("--json", type=str, help="JSON Object to send to server")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080",
        help="URL to send JSON Object to",
    )
    parser.add_argument("--endpoint", type=str, help="Endpoint to send JSON Object to")
    args = parser.parse_args()
    send_json(args)
