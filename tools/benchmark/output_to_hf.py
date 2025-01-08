import argparse
import os
import json
import datasets

def upload(args):
    output_dir = args.output_dir
    dirs = os.listdir(output_dir)
    rows = []
    for dir in dirs:
        path = os.path.join(output_dir, dir)
        with open(path, 'r') as f:
            json_data = json.loads(f.readline())
            json_data['device'] = args.device
            del json_data['server_args']['json_model_override_args']
            rows.append(json_data)
            # print(json_data)
    dataset = datasets.Dataset.from_list(rows)
    print(dataset)
    dataset.push_to_hub("xiaozheyao/btc_perf", config_name=f"{args.device} | {rows[0]['client_args']['model']}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload to HF")
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, required=True, help='Device used for benchmarking')
    args = parser.parse_args()
    upload(args)