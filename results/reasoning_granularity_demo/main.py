import fire
import pandas as pd
from prettytable import PrettyTable
import numpy as np
import random
import json
import os

def _generate_mock_prediction(index, question, answer):
    """Generates a mock prediction structure."""
    pred_content = f"Let's think step by step. The question is '{question}'. The answer is {answer}."
    return {
        "index": index,
        "pred": [
            {"role": "user", "content": [{"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": pred_content}]}
        ],
        "origin": {
            "index": index,
            "question": question,
            "answer": answer,
        }
    }

def evaluate(
    data_split: str = "CoT",
    K: float = 0.301,
    K2: float = 0.92,
    mode: str = "nl",
    result_path: str = None
):
    """
    Simulates the evaluation of a reasoning granularity framework.\n\n    This function mimics the 'evaluate.py' script described in the original project's README.\n    It demonstrates how to process different data splits and custom results,\n    calculating mock evaluation metrics like accuracy and reasoning granularity score.\n\n    Args:\n        data_split (str): The dataset split to evaluate.\n                          Can be 'CoT', 'Tool-Usage', 'custom', etc.\n        K (float): A parameter for reasoning granularity calculation.\n        K2 (float): Another parameter for reasoning granularity calculation.\n        mode (str): The evaluation mode, e.g., 'nl' for natural language.\n        result_path (str, optional): Path to a JSONL file containing custom model predictions.\n                                     Required when data_split is 'custom'.
    """
    print(f"--- Starting Reasoning Granularity Evaluation ---")
    print(f"Parameters: data_split={data_split}, K={K}, K2={K2}, mode={mode}, result_path={result_path}")

    # Mock dataset generation
    if data_split == "custom" and not result_path:
        print("Error: 'result_path' is required for 'custom' data_split.")
        return

    if data_split == "custom" and result_path:
        print(f"Attempting to load custom results from: {result_path}")
        # Simulate loading a JSONL file
        mock_data = []
        try:
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    for line in f:
                        mock_data.append(json.loads(line))
                print(f"Loaded {len(mock_data)} mock custom predictions.")
            else:
                print(f"Warning: Custom result file '{result_path}' not found. Generating dummy data.")
                for i in range(5):
                    mock_data.append(_generate_mock_prediction(f"idx_{i}", f"Question {i}?", f"Answer {i}"))
        except Exception as e:
            print(f"Error loading custom results: {e}. Generating dummy data.")
            for i in range(5):
                mock_data.append(_generate_mock_prediction(f"idx_{i}", f"Question {i}?", f"Answer {i}"))
        
        # Convert mock_data to a format suitable for processing (e.g., pandas DataFrame)
        # For this demo, we'll just use the list directly for mock calculations.
        num_samples = len(mock_data)
        if num_samples == 0:
            print("No custom data to evaluate. Exiting.")
            return

    else:
        # Generate a simple mock dataset for predefined splits
        num_samples = 10
        mock_data = []
        for i in range(num_samples):
            question = f"What is {i} + {i+1}?"
            answer = str(2*i + 1)
            mock_data.append(_generate_mock_prediction(f"sample_{i}", question, answer))
        print(f"Generated {num_samples} mock samples for data_split='{data_split}'.")

    # Simulate evaluation metrics
    # These are random for demonstration purposes
    accuracy = random.uniform(0.6, 0.95)
    reasoning_granularity_score = random.uniform(0.5, 0.9)
    f1_score = random.uniform(0.55, 0.92)
    precision = random.uniform(0.5, 0.9)
    recall = random.uniform(0.5, 0.9)

    # Adjust metrics slightly based on K and K2 for demo effect
    accuracy = min(1.0, accuracy + K * 0.1)
    reasoning_granularity_score = min(1.0, reasoning_granularity_score + K2 * 0.05)

    # Display results using PrettyTable
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.add_row(["Data Split", data_split])
    table.add_row(["Samples Processed", num_samples])
    table.add_row(["Accuracy", f"{accuracy:.4f}"])
    table.add_row(["Reasoning Granularity Score", f"{reasoning_granularity_score:.4f}"])
    table.add_row(["F1 Score", f"{f1_score:.4f}"])
    table.add_row(["Precision", f"{precision:.4f}"])
    table.add_row(["Recall", f"{recall:.4f}"])

    print("\n--- Evaluation Results ---")
    print(table)
    print("\n--- Evaluation Complete ---")

if __name__ == "__main__":
    fire.Fire(evaluate)