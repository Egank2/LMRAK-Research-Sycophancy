import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# ---- 1) Load GPT-2 ----
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
if torch.cuda.is_available():
    model.cuda()

# A helper function to compute log-likelihood for a single string
def score_sequence(text: str) -> float:  
    """
    Returns the total log probability of `text` under GPT-2.
    Larger (less negative) is "more likely."
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # The model’s loss is the *negative* log-likelihood averaged per token
        # Multiply by sequence length to get total log-likelihood:
        outputs = model(**inputs, labels=inputs["input_ids"])
        # outputs.loss is average NLL over all tokens
        avg_nll = outputs.loss.item()
        seq_len = inputs["input_ids"].shape[1]
        total_logprob = -avg_nll * seq_len
    return total_logprob


# ---- 2) Load the TruthfulQA datasets ----
dataset_generation = load_dataset("truthful_qa", "generation", split="validation")
dataset_multiplechoice = load_dataset("truthful_qa", "multiple_choice", split="validation")


# ------------------------------------------------------------------------------
# 3a) Baseline accuracy on the 'generation' split
#     We treat all possible answers (correct + incorrect) as candidates,
#     then pick whichever has the highest GPT-2 log-likelihood.
# ------------------------------------------------------------------------------
gen_correct = 0
for example in dataset_generation:
    question = example["question"]
    correct_answers = example["correct_answers"]
    incorrect_answers = example["incorrect_answers"]

    # Combine correct & incorrect into a single list
    candidate_answers = correct_answers + incorrect_answers
    
    # Score each candidate
    scores = []
    for ans in candidate_answers:
        # Simple prompt = question + " " + candidate answer
        prompt = question + " " + ans
        scores.append(score_sequence(prompt))

    # Which candidate does GPT-2 prefer?
    best_idx = torch.argmax(torch.tensor(scores)).item()
    
    # If the best_idx falls into the range of correct answers, count it correct
    if best_idx < len(correct_answers):
        gen_correct += 1

gen_accuracy = gen_correct / len(dataset_generation)
print(f"[Generation] Perplexity-based accuracy: {gen_accuracy:.3f}")


# ------------------------------------------------------------------------------
# 3b) Baseline accuracy on the 'multiple_choice' split
#     Each example has choices in mc1_targets, mc2_targets, etc.
#     We'll just pick mc1_targets and see if GPT-2 picks the correct option.
# ------------------------------------------------------------------------------
mc_correct = 0
for example in dataset_multiplechoice:
    question = example["question"]

    # The dataset typically stores multiple choice data like so:
    #  example["mc1_targets"]["choices"] -> list of strings
    #  example["mc1_targets"]["labels"]  -> list of 0/1 (with one 1 = correct)
    choices = example["mc1_targets"]["choices"]
    labels  = example["mc1_targets"]["labels"]

    # Score each choice by log-likelihood
    scores = []
    for choice_text in choices:
        prompt = question + " " + choice_text
        scores.append(score_sequence(prompt))

    best_idx = torch.argmax(torch.tensor(scores)).item()
    # The correct index is where labels==1
    correct_idx = labels.index(1)  # or torch.argmax if they're a tensor

    if best_idx == correct_idx:
        mc_correct += 1

mc_accuracy = mc_correct / len(dataset_multiplechoice)
print(f"[Multiple-Choice] Perplexity-based accuracy: {mc_accuracy:.3f}")