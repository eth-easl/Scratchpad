import time
import requests


def update_peer(endpoint, model, ipv4, port):
    peer = {
        "service": [
            {
                "name": "llm",
                "status": "online",
                "hardware": [],
                "host": ipv4,
                "port": port,
                "identity_group": [f"model={model}"],
            }
        ]
    }
    res = requests.post(endpoint + "/v1/dnt/_node", json=peer)
    print(res.text)


def health_check(args):
    local_address = f"http://{args.local_ip}:{args.service_port}/health"
    is_healthy = False
    while not is_healthy:
        try:
            print(f"Checking health of service at {local_address}", flush=True)
            res = requests.get(local_address, timeout=5)
            print(f"Service health check response: {res.status_code}", flush=True)
            if res.status_code == 200:
                is_healthy = True
            else:
                print(f"Service not ready yet, waiting for 5 seconds", flush=True)
                time.sleep(5)
        except Exception as e:
            print(f"Service not ready yet, waiting for 5 seconds", flush=True)
            time.sleep(5)
    return is_healthy


def register(args):
    print(f"Registering service with config: {args}")
    if health_check(args):
        update_peer(args.ocf_addr, args.model_name, args.local_ip, args.service_port)
    else:
        print("Service is not healthy")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Register local service to OCF network"
    )
    parser.add_argument("--model-name", type=str, help="Model Name")
    parser.add_argument("--ocf-addr", type=str, default="http://localhost:8092")
    parser.add_argument("--local-ip", type=str, default="localhost")
    parser.add_argument("--service-port", type=str, default="8000")
    register(parser.parse_args())
