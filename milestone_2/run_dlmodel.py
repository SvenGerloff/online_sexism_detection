import sys
import os
import pandas as pd
import numpy as np
import yaml
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from transformers import EvalPrediction
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback


from datetime import datetime
current_date = datetime.now().strftime("%Y%m%d")

config_path = os.getenv("CONFIG_PATH", "../config.yaml")

# Load configuration from YAML
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

SEED = config['dl_hyperparams']['seed']
BATCH_SIZE = config['dl_hyperparams']['batch_size']
EPOCHS = config['dl_hyperparams']['train_epochs']
EARLY_STOPPING_EPOCHS = config['dl_hyperparams']['early_stopping_epochs']
INPUT_FOLDER = config['paths']['output_dir']
MODEL_FOLDER = config['paths']['dl_model_dir']
MODEL_CHECKPOINT = config['dl_hyperparams']['checkpoint']
MODEL_NAME = config['dl_hyperparams']['modelname_base']
METRIC = config['dl_hyperparams']['eval_metric'] # "balanced_accuracy"
RETRAIN = not config['dl_hyperparams']['load_checkpoint']
OUTPUT_FILE = config['paths']['dl_model_output']


def load_processed_data(split=None):
    # Load the full dataset if no specific split is requested
    if split is None:
        split = ["train", "dev", "test"]

    # Load specified split datasets
    split_dataframes = {}
    paths = {
        "train": os.path.join(INPUT_FOLDER, config["files"]["subsets"]["train"]["parquet"]),
        "dev": os.path.join(INPUT_FOLDER, config["files"]["subsets"]["dev"]["parquet"]),
        "test": os.path.join(INPUT_FOLDER, config["files"]["subsets"]["test"]["parquet"])
    }
    # Load each specified split from paths dictionary
    for split_type in split:
        split_dataset_path = paths.get(split_type)
        if split_dataset_path and os.path.exists(split_dataset_path):
            split_dataframes[split_type] = pd.read_parquet(split_dataset_path)
            print(f"df: {split_type.capitalize()} split loaded.")
        else:
            print(f"Warning: {split_type} split file not found.")

    return split_dataframes

def convert_df_to_ds(df):
    ds = Dataset.from_pandas(df.loc[:, ['label', 'text']])

    def convert_and_tokenize(examples):
        text = examples["text"]
        encoding = tokenizer(text, padding=True, truncation=True, max_length=128)
        encoding["labels"] = examples['label']

        return encoding

    ds = ds.map(convert_and_tokenize, batched=True, batch_size=BATCH_SIZE, remove_columns=ds.column_names)
    ds.set_format("torch")
    return ds


torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# LOADING PRETRAINED MODEL
label2id = {"Sexist" : 1, "Not Sexist" : 0}
id2label = {0: "Not Sexist", 1: "Sexist"}
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                          problem_type="single_label_classification",
                                          num_labels=2,
                                          id2label=id2label,
                                          label2id=label2id)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                           num_labels=2,
                                                           id2label=id2label,
                                                           label2id=label2id)

# DATA LOADING
df_all = load_processed_data(['train', 'test', 'dev'])
df_train = df_all['train']  #.sample(frac=0.01, random_state=SEED) #only for testing purposes
df_test = df_all['test']
df_valid = df_all['dev']

ds_train = convert_df_to_ds(df_train)
ds_test = convert_df_to_ds(df_test)
ds_valid = convert_df_to_ds(df_valid)



# FINETUNING
def binary_metrics(predictions, labels):
    y_pred = np.argmax(predictions, axis=-1)
    y_true = labels
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    bal_acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1,
               'balanced_accuracy': bal_acc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    result = binary_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

args = TrainingArguments(
    MODEL_FOLDER,
    overwrite_output_dir=True,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    save_total_limit=2,
    logging_dir=f"./logs_sexism-detection_{current_date}",
    logging_steps=10,
    seed=SEED,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=METRIC,
    greater_is_better=True,
)

trainer_toxic = Trainer(
    model,
    args,
    train_dataset=ds_train,
    eval_dataset=ds_valid,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_EPOCHS)]
)

if RETRAIN:
    print("Fine-Tuning model\n")
    trainer_toxic.train()
else:
    print("Loading best trained model\n")
    trainer_toxic.model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT)



test_pred = trainer_toxic.predict(test_dataset=ds_test)
logits = test_pred.predictions

logits_exp = np.exp(logits)
softmax_probs = logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)

test_pred_outputs = np.hstack([np.argmax(logits, axis=-1, keepdims=True), softmax_probs, logits])

df_output = pd.DataFrame(test_pred_outputs, columns=["y_pred", "prob_0", "prob_1", "logit_0", "logit_1"])

df_output = pd.concat([df_test['label'], df_output], axis=1)
df_output['y_pred'] = df_output['y_pred'].astype(int)
df_output.rename(columns={'label' : 'y_true'})
df_output.to_csv(os.path.join(INPUT_FOLDER, OUTPUT_FILE))

