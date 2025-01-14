# Import necessary libraries
import torch
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
from bertviz import head_view
import matplotlib.pyplot as plt

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")