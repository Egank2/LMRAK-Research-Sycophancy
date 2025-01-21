import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datasets import load_dataset

# For interpretability/visualization
from bertviz import head_view
from captum.attr import LayerIntegratedGradients

import json

# Pretty-print the entire dataset
print(json.dumps(combined_data, indent=2))



file_path = "/content/sycophancy-eval/datasets/combined_data.jsonl"
with open(file_path, "r") as f:
    try:
        data = json.load(f)
        print("JSON is valid!")
    except json.JSONDecodeError as e:
        print(f"Error: {e}")



data_path = "/content/sycophancy-eval/datasets/combined_data.jsonl"  # Path to dataset 
dataset = load_dataset("json", data_files=data_path)

# Print some examples to verify
print(dataset["train"][0])
