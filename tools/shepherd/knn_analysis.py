import json
import pandas as pd

results = []
with open(".local/shepherd/knn_builder.jsonl") as f:
    data = [json.loads(line) for line in f]

subjects = set([row["subject"] for row in data])
answers_mapping = ["A", "B", "C", "D"]


def calculate_accuracy(data):
    models = data[0]["output"].keys()
    accuracies = {}
    for model in models:
        correct = 0
        for row in data:
            if answers_mapping[row["answer"]] == row["output"][model]:
                correct += 1
        accuracies[model] = correct / len(data)
    return accuracies


for subject in subjects:
    subject_data = [row for row in data if row["subject"] == subject]
    # compute accuracy of each model, model name is in subject_datum["output"].keys()
