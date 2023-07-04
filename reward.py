from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from trl.trainer.trainer import Trainer
from transformers.utils import PaddingStrategy
from trl.core import print_trainable_parameters
import os 
# os.environ["WANDB_DISABLED"] = "true"
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    resume_from_checkpoint: Optional[bool] = field(
        default=False, metadata={"help": "If you want to resume training where it left off."}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=16)
    per_device_eval_batch_size: Optional[int] = field(default=16)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[int] = field(default=2e-5)
    weight_decay: Optional[int] = field(default=0.001)
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default="5", metadata={"help": "The number of training epochs for the reward model. OpenAI used 5."}
    )
    train_file: Optional[str] = field(
        default="",
    )
    dev_file: Optional[str] = field(
        default="",
    )
    output_dir: Optional[str] = field(
        default="",
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
# Load the human comparisons dataset for tuning the reward model.
data_files = {}
data_files["train"] = script_args.train_file
data_files["validation"] = script_args.dev_file
ds = load_dataset(
    "json",
    data_files=data_files,
)
# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
training_args = TrainingArguments(
    output_dir= script_args.output_dir,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    deepspeed=script_args.deepspeed,
    # local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    report_to="wandb"
)

# Load the value-head model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(script_args.model_name, num_labels=1)
# print_trainable_parameters(model)
# Need to do this for gpt2, because it doesn't have an official pad token.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Turn the dataset into pairs of post + summaries, where text_j is the preferred post + summary and text_k is the other.
def turn_into_text_classification_format(examples):
    new_examples = {"text_j": [], "text_k": []}
    prompt = "请对以下微博做出评论:"
    for weibo, text_j, text_k in zip(examples["weibo"], examples["text_j"], examples["text_k"]):
        new_examples["text_j"].append(
            text_j + " " + tokenizer.bos_token + " " +  weibo
        )
        new_examples["text_k"].append(
            text_k + " " + tokenizer.bos_token + " " +  weibo
        )
    return new_examples
original_columns = ds["train"].column_names
print("ds[train]",ds["train"][0])
print("ds[eval]",ds["validation"][0])
print("original_columns",ds,len(ds),type(ds))
ds = ds.map(turn_into_text_classification_format, batched=True, num_proc=1, remove_columns=original_columns)
print("ds[train]_after",ds["train"][0])
print("ds[eval]_after",ds["validation"][0])
print("original_columns_after",ds,len(ds),type(ds))
# Tokenize the dataset.

def preprocess_function(examples):
    tokenized_j = tokenizer(examples["text_j"], truncation=True, max_length=160)
    tokenized_k = tokenizer(examples["text_k"], truncation=True, max_length=160)
    return {
        "input_ids_j": tokenized_j["input_ids"],
        "attention_mask_j": tokenized_j["attention_mask"],
        "input_ids_k": tokenized_k["input_ids"],
        "attention_mask_k": tokenized_k["attention_mask"],
    }
tokenized_ds = ds.map(preprocess_function, batched=True, num_proc=1, remove_columns=["text_j", "text_k"])


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append({"input_ids": feature["input_ids_j"], "attention_mask": feature["attention_mask_j"]})
            features_k.append({"input_ids": feature["input_ids_k"], "attention_mask": feature["attention_mask_k"]})
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


class RewardTrainer(Trainer):
    # Define how to compute the reward loss.
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        print("rewards_j",rewards_j)
        print("rewards_k",rewards_k)
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.evaluate()
loss = trainer.train(script_args.resume_from_checkpoint)
print("loss",loss)

