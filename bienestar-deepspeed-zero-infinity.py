# %%
# confirm I'm in the lab and have access to the resources
# 3 Nvidia A100 GPUs @ 80 GB each
# !nvidia-smi

# %%
# pip install the required packages
#import sys
# !{sys.executable} -m pip install numpy pandas matplotlib scikit-learn datasets torch accelerate evaluate transformers[deepspeed]

# %%
# Run the following command to see if bigscience fits
#import sys
# !{sys.executable} -c 'from transformers import AutoModel; \
# from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
# model = AutoModel.from_pretrained("bigscience/T0_3B"); \
# estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)'

# %%
# make sure to do a `pip install -r requirements.txt` prior to running through this notebook.
import numpy as np
import pandas as pd
import torch
torch.cuda.is_available()

# %%
#from datasets import load_dataset
#from sklearn.model_selection import train_test_split

#ds = load_dataset("avacaondata/bioasq22-es")
#dsTrainTest_80_20 = ds['train'].train_test_split(test_size=0.2)
#dsTrainTest_80_20

# Spanish QA.  May need to use if question and answering understanding is poor for the model.
#data_files = {"train": "datasets/squad-train-v1.1_clean-spanish.json", "test": "datasets/squad-dev-v1.1_clean-spanish.json"}
#raw_datasets = load_dataset("json", data_files=data_files)
#raw_datasets

from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DataCollatorWithPadding
from collections import defaultdict

ds = load_dataset("avacaondata/bioasq22-es")
dsTrainTest_80_20 = ds['train'].train_test_split(test_size=0.2, seed=42) # added seed for reproducibility

# %%
print ("Train Sample")
print("Context: ", dsTrainTest_80_20["train"][5]['context'])
print("Question: ", dsTrainTest_80_20["train"][5]['question'])
print("Answer: ", dsTrainTest_80_20["train"][5]['answers'])

print("Test Sample")
print(dsTrainTest_80_20["test"][5]['context'])

# %%
#from transformers import AutoTokenizer

#model_checkpoint = "bert-base-multilingual-cased"
#tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#tokenizer.is_fast # make sure that the model is supported by the fast tokenizer by checking if its fast
model_checkpoint = "bert-base-multilingual-cased"  # Or your specific checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)


# %%
context = dsTrainTest_80_20["train"][0]['context']
question = dsTrainTest_80_20["train"][0]['question']
inputs = tokenizer(question, context)
tokenizer.decode(inputs["input_ids"])

# %%
max_length = 384
stride = 128

# %%
def preprocess_training_examples(examples):
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
                 idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# %%
def preprocess_testing_examples(examples):
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

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

# %%
train_dataset = dsTrainTest_80_20["train"].map(
    preprocess_training_examples,
    batched=True,
                                remove_columns=dsTrainTest_80_20["train"].column_names, #remove here
)

# %%
eval_dataset = dsTrainTest_80_20["test"].map(
    preprocess_testing_examples,
    batched=True,
    remove_columns=dsTrainTest_80_20["test"].column_names, #remove here
)

# %%
#training_args = TrainingArguments(
#    output_dir="mbert-fine-tune",  # Choose a directory
#    evaluation_strategy="epoch",
#    learning_rate=2e-5,
#    per_device_train_batch_size=16,  # Adjust as needed
#    per_device_eval_batch_size=16,   # Adjust as needed
#    num_train_epochs=3,             # Adjust as needed
#    weight_decay=0.01,
#    fp16=True,  # Use mixed precision (if your GPU supports it)
#    #push_to_hub=True, # Uncomment if you want to push to the Hub
#    #hub_model_id="your_username/your_repo_name",  # If pushing to the Hub
#    save_strategy="epoch", #save checkpoints at the end of each epoch
#    load_best_model_at_end = True, #load the best model
#    metric_for_best_model="exact_match"
#)

# %%
# Use DataCollatorWithPadding (important for variable-length sequences)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
eval_dataset

# %%
import evaluate  # Import the evaluate library
from collections import defaultdict

def compute_metrics(eval_preds):
    start_logits, end_logits = eval_preds.predictions
    # Use eval_preds.dataset to access the processed dataset
    examples = eval_preds.dataset
    example_ids = examples["example_id"]
    offset_mapping = examples["offset_mapping"]

    # VERY IMPORTANT: Get the original, UNPROCESSED dataset for the references.
    raw_eval_dataset = dsTrainTest_80_20["test"]  # Use the original dataset

    sample_map = defaultdict(list)
    for i, example_id in enumerate(example_ids):
        sample_map[example_id].append(i)

    n_best = 20
    max_answer_length = 30

    predicted_answers = []
    for example in examples:  # Iterate through *processed* examples
        example_id = example["example_id"]

        # Find the corresponding raw example using example_id
        raw_example_indices = [i for i, ex in enumerate(raw_eval_dataset) if ex["id"] == example_id]
        if not raw_example_indices:
            print(f"WARNING: Could not find original example with ID {example_id}")
            continue  # Skip this example if no matching raw example is found

        raw_example_index = raw_example_indices[0]  # Should only be one match
        raw_example = raw_eval_dataset[raw_example_index]
        context = raw_example["context"] #use the original context

        for feature_index in sample_map[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = offset_mapping[feature_index]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                        if (end_index < start_index) or (end_index - start_index + 1 > max_answer_length):
                        continue
                    if start_index >= len(offsets) or end_index >= len(offsets):
                        continue

                    start_char = offsets[start_index][0]
                    end_char = offsets[end_index][1]
                    predicted_answers.append(
                        {
                            "id": example_id,  # Use the example_id
                            "prediction_text": context[start_char:end_char],  # Extract text
                        }
                    )

    formatted_predictions = [{"id": answer["id"], "prediction_text": answer["prediction_text"]} for answer in predicted_answers]

    # Get references directly from the *original* dataset
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in raw_eval_dataset]
    references = [{"id": ref["id"], "answers": {"answer_start": [ans["answer_start"][0]], "text": [ans["text"][0]]}} for ref in references]


    # Deduplicate references (important when using a sliding window)
    unique_references = {}
    for ref in references:
        unique_references[ref["id"]] = ref["answers"]
    final_references = [{"id": id, "answers": answers} for id, answers in unique_references.items()]

     # Deduplicate predictions (important when using a sliding window)
    unique_predictions = {}
    for prediction in formatted_predictions:
      unique_predictions[prediction["id"]] = prediction["prediction_text"]

    final_predictions = [{"id":id, "prediction_text": pred_text} for id, pred_text in unique_predictions.items()]

    results = metric.compute(predictions=final_predictions, references=final_references)

    return {
        "eval_f1": results["f1"],
        "eval_exact_match": results["exact_match"],
    }


training_args = TrainingArguments(
    output_dir="mbert-fine-tune-deepspeed",  # Choose a directory
    #evaluation_strategy="epoch",
    #learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    #weight_decay=0.01,  # set in deepSpeed config
    fp16=True,  # Use mixed precision (if your GPU supports it)
    #push_to_hub=True, # Uncomment if you want to push to the Hub
    #hub_model_id="your_username/your_repo_name",  # If pushing to the Hub
    save_strategy="epoch", #save checkpoints at the end of each epoch
    save_total_limit=2,  # limits the number of checkpoints
    #load_best_model_at_end = True, #load the best model
    #metric_for_best_model="f1"
    gradient_accumulation_steps=4,
    deepspeed="ds_config.json",
    dataloader_num_workers=12,
    dataloader_pin_memory=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # Use the QA compute_metrics
    data_collator=data_collator,
)

#training_args


#results = trainer.evaluate()
#print(results)

# %%
# Single GPU training without DeepSpeed configuration
#trainer.train()

# %%
#results = trainer.evaluate()
#print(results)

# %%
# Configure DeepSpeed for parallelization# update training arguments with DeepSpeed configuration
# training arguments for deepspeed should be in nested dic structure
# examples in https://github.com/deepspeedai/DeepSpeedExamples/
#training_args.deepspeed = {
#                           "train_batch_size": 24,
#                           "train_micro_batch_size_per_gpu": 3,
#                           "steps_per_print": 10,
#                           "optimizer": {
#                               "type": "Adam",
#                               "params": {
#                                   "lr": 3e-5,
#                                   "weight_decay": 0.0,
#                                   "bias_correction": False,
#                               }
#                           },
#                           "gradient_clipping": 1.0,
#                           "fp16": {
#                               "enabled": False
#                           },
#                           #"gradient_accumulation_steps": "auto",
#                           "checkpoint": {
#                               "load_universal": True
#                           }
#}

#training_args
if __name__ == "__main__":
    # Run on 3 GPU
    train_result = trainer.train()
    trainer.save_model()
    metrics = trainer.evaluate()
    print(metrics)