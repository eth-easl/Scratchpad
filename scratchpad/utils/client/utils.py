import time
import requests


def wait_until_ready(host: str, port: str):
    while True:
        try:
            res = requests.get(f"http://{host}:{port}/health")
            if res.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
