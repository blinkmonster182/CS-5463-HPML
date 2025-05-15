# confirm I'm in the lab and have access to the resources
# 3 Nvidia A100 GPUs @ 80 GB each
!nvidia-smi

# pip install the required packages
import sys
!{sys.executable} -m pip install numpy pandas matplotlib scikit-learn datasets torch accelerate evaluate transformers[deepspeed] tqdm # Added tqdm for progress bar

# Run the following command to see if bigscience fits
#import sys
!{sys.executable} -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("bigscience/T0_3B"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)'

# make sure to do a `pip install -r requirements.txt` prior to running through this notebook.
from transformers.utils import logging as transformers_logging
transformers_logging.set_verbosity_info()
logger = transformers_logging.get_logger(__name__)

import numpy as np
import pandas as pd
import torch
from functools import partial
torch.cuda.is_available()

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DataCollatorWithPadding, EvalPrediction
from collections import defaultdict
import evaluate # Import evaluate library
from tqdm.auto import tqdm # For progress bars
from multiprocessing import Pool, cpu_count

# --- 1. Load Data ---
ds = load_dataset("avacaondata/bioasq22-es")
dsTrainTest_80_20 = ds['train'].train_test_split(test_size=0.2, seed=42) # added seed for reproducibility

# Make the raw test dataset globally accessible (or pass it appropriately)
raw_test_dataset = dsTrainTest_80_20["test"]

print ("Train Sample")
print("Context: ", dsTrainTest_80_20["train"][5]['context'])
print("Question: ", dsTrainTest_80_20["train"][5]['question'])
print("Answer: ", dsTrainTest_80_20["train"][5]['answers'])

print("Test Sample")
print(raw_test_dataset[5]['context']) # Use the global variable

model_checkpoint = "bert-base-multilingual-cased"  # Or your specific checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

context = dsTrainTest_80_20["train"][0]['context']
question = dsTrainTest_80_20["train"][0]['question']
inputs = tokenizer(question, context)
tokenizer.decode(inputs["input_ids"])

max_length = 384
stride = 128

# --- Preprocessing Functions (Keep as they are) ---
def preprocess_training_examples(examples, tokenizer, max_length, stride):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        # Handle cases where answer['text'] or answer['answer_start'] might be empty.
        if not answer['text'] or not answer['answer_start']:
          start_positions.append(0)
          end_positions.append(0)
          continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_test_examples(examples, tokenizer, max_length, stride):
    """Process the validation split of the SQuAD dataset.

    Process the training split of the SQuAD dataset to include the unique ID of each row,
    the tokenized questions and context, as well as the start and end positions of the answer
    within the context.

    Args:
        examples: A row from the dataset containing an example.
        tokenizer: The BERT tokenizer to be used.
        max_length: The maximum length of the input sequence. If exceeded, truncate the second
            sentence of a pair (or a batch of pairs) to fit within the limit.
        stride: The number of tokens to retain from the end of a truncated sequence, allowing
            for overlap between truncated and overflowing sequences.

    Returns:
        The processed example.
    """
    # Tokenize the questions and context sequences
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
      questions,
      examples["context"],
      truncation="only_second",
      padding="max_length",
      stride=stride,
      max_length=max_length,
      return_offsets_mapping=True,
      return_overflowing_tokens=True,
    )

    example_ids = []
    answers = examples["answers"]
    offset_mapping = inputs["offset_mapping"]
    sample_map = inputs.pop("overflow_to_sample_mapping")

    start_positions = []
    end_positions = []

    # find the start and end positions of the answer within the context
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # if the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["example_id"] = example_ids  # keep the unique ID of the example
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


preprocess_train_data = partial(preprocess_training_examples, tokenizer=tokenizer, max_length=384, stride=128)
processed_train_data = dsTrainTest_80_20["train"].map(preprocess_train_data, batched=True, remove_columns=dsTrainTest_80_20["train"].column_names)
processed_train_data

preprocess_test_data = partial(
    preprocess_test_examples, tokenizer=tokenizer, max_length=384, stride=128)
processed_test_data = dsTrainTest_80_20["test"].map(preprocess_test_data, batched=True, remove_columns=dsTrainTest_80_20["test"].column_names)
processed_test_data

# Use DataCollatorWithPadding (important for variable-length sequences)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


import numpy as np
import evaluate  # Import the evaluate library
import collections


# Ensure metric is loaded globally or at the start of the script
f1_metric = evaluate.load("f1")
squad_metric = evaluate.load("squad")


def process_single_example(example, example_to_features, features, start_logits, end_logits, n_best, max_answer_length):
    # --- This function remains exactly the same as provided in the previous answer ---
    example_id = example["id"]
    context = example["context"]
    candidate_answers = []

    # Get predictions
    feature_indices = example_to_features.get(example_id)
    if feature_indices:
        for feature_index in feature_indices:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            # Access feature data - ensure features is accessible by index
            try:
                 feature_data = features[feature_index]
                 offsets = feature_data.get("offset_mapping")
            except IndexError:
                 print(f"Warning: Feature index {feature_index} out of bounds for features length {len(features)}")
                 continue # Skip this feature if index is invalid

            if offsets is None: continue # Skip if no offsets

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offsets) or end_index >= len(offsets) or offsets[start_index] is None or offsets[end_index] is None:
                         continue
                    # Minor fix: Check offset tuple contents, not the tuples themselves for None
                    if offsets[start_index][0] > offsets[end_index][1]:
                            continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                         continue
                    try:
                         answer_text = context[offsets[start_index][0] : offsets[end_index][1]]
                    except IndexError: continue

                    candidate_answers.append({
                        "text": answer_text,
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    })

    best_prediction_text = ""
    if candidate_answers:
        best_answer = max(candidate_answers, key=lambda x: x["logit_score"])
        best_prediction_text = best_answer["text"]

    formatted_prediction = {"id": str(example_id), "prediction_text": best_prediction_text}

    # Format theoretical answers
    original_answers = example.get("answers", {})
    texts = original_answers.get("text", [])
    starts = original_answers.get("answer_start", [])
    squad_formatted_answers = []
    if isinstance(texts, list) and isinstance(starts, list) and len(texts) == len(starts):
         for i in range(len(texts)):
              if isinstance(texts[i], str) and isinstance(starts[i], int):
                  squad_formatted_answers.append({"text": texts[i], "answer_start": starts[i]})

    formatted_reference = {"id": str(example_id), "answers": squad_formatted_answers}

    return formatted_prediction, formatted_reference

# --- Main compute_metrics function for Trainer ---
def compute_metrics_mp(p: EvalPrediction, n_best=20, max_answer_length=50):
    # Access global datasets (defined earlier in the script)
    # Ensure these are the correct datasets corresponding to the predictions
    global processed_test_data # The features dataset used for evaluation
    global raw_test_dataset    # The raw examples dataset used for evaluation

    # Extract predictions
    start_logits, end_logits = p.predictions

    # Assign features and examples from globals (make sure they align with predictions)
    # This assumes evaluate() is always called on the same test set defined globally
    features = processed_test_data
    examples = dsTrainTest_80_20["test"]

    # Build mapping (usually fast)
    example_to_features = collections.defaultdict(list)
    # print("Mapping features to examples...") # Optional print
    if "example_id" not in features.column_names:
        raise ValueError("Features dataset must contain 'example_id' column.")
    # Iterate through features['example_id'] directly if it's a dataset object
    try:
        for idx, feature_example_id in enumerate(features["example_id"]):
            example_to_features[feature_example_id].append(idx)
    except TypeError:
        # Fallback if features isn't a dataset object but maybe a list of dicts
        print("Warning: Features might not be a Dataset object. Iterating differently.")
        for idx, feature in enumerate(features):
             example_to_features[feature["example_id"]].append(idx)


    # Prepare partial function for Pool.map
    # Convert examples Dataset to list for Pool.map compatibility
    examples_list = list(examples)

    # Pass necessary data to the worker function
    process_func = partial(
        process_single_example,
        example_to_features=example_to_features,
        features=features, # Pass the features dataset/list
        start_logits=start_logits, # Pass all logits
        end_logits=end_logits,
        n_best=n_best,
        max_answer_length=max_answer_length
    )

    num_workers = max(1, cpu_count() // 2) # Adjust number of workers for metric processing

    predicted_answers = []
    theoretical_answers = []

    # Use Pool.imap with tqdm for progress bar
    with Pool(num_workers) as pool:
         results = list(tqdm(pool.imap(process_func, examples_list, chunksize=32), total=len(examples_list), desc="Post-processing")) # Adjust chunksize as needed

    # Unzip results
    for pred, ref in results:
         predicted_answers.append(pred)
         theoretical_answers.append(ref)

    # print("Computing final metric...") # Optional print
    if not predicted_answers or not theoretical_answers: return {}

    return squad_metric.compute(predictions=predicted_answers, references=theoretical_answers)


# --- Training Arguments and Trainer Setup ---
training_args = TrainingArguments(
    output_dir="mbert-fine-tune-deepspeed",  # Choose a directory
    eval_strategy="epoch", # Evaluate at the end of each epoch to use compute_metrics
    #learning_rate=2e-5, # Set in deepSpeed config ("auto")
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16, # Adjust eval batch size based on GPU memory
    num_train_epochs=3,
    #weight_decay=0.01,  # set in deepSpeed config ("auto")
    fp16=True,  # Use mixed precision (if your GPU supports it)
    #push_to_hub=True, # Uncomment if you want to push to the Hub
    #hub_model_id="your_username/your_repo_name",  # If pushing to the Hub
    save_strategy="epoch", #save checkpoints at the end of each epoch
    save_total_limit=2,  # limits the number of checkpoints
    load_best_model_at_end = True, #load the best model based on metric
    metric_for_best_model="f1",
    gradient_accumulation_steps=4, # Set in deepSpeed config ("auto")
    deepspeed="ds_config.json",
    dataloader_num_workers=12,
    dataloader_pin_memory=True,
    report_to="none", # Optional: disable wandb/tensorboard reporting if not needed
    logging_steps=50,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train_data,
    eval_dataset=processed_test_data, # Use the processed dataset for evaluation
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_mp,  # Use the QA compute_metrics function
    data_collator=data_collator,
)

# --- Run Training and Evaluation ---
if __name__ == "__main__":
    print("Starting training...")
    train_result = trainer.train()
    print("Training finished.")

    print("Saving final model...")
    trainer.save_model() # Saves the tokenizer too
    trainer.save_state()
    print("Model saved.")

    print("Evaluating final model...")
    metrics = trainer.evaluate() # This will call compute_metrics
    print("Evaluation finished.")

    print("Evaluation Metrics:")
    print(metrics) # Print the computed metrics (f1, exact_match)
