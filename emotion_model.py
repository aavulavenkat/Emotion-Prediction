import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import config
from sklearn.metrics import accuracy_score
import os
import json


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions)
    }


def train_model():
    # Load the GoEmotions dataset
    dataset = load_dataset("google-research-datasets/go_emotions")

    # Save original label names BEFORE simplification
    config.LABELS = dataset["train"].features["labels"].feature.names

    # Convert multi-label to single-label (pick the first label)
    def simplify_labels(example):
        if example["labels"]:
            example["label"] = example["labels"][0]
        else:
            example["label"] = 0  # fallback
        return example

    dataset = dataset.map(simplify_labels)

    # Remove original "labels" column to avoid nested lists
    dataset = dataset.remove_columns("labels")

    # Optional: limit dataset size
    dataset["train"] = dataset["train"].select(range(10000))
    dataset["validation"] = dataset["validation"].select(range(2000))
    dataset["test"] = dataset["test"].select(range(2000))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    dataset = dataset.map(tokenize, batched=True)

    # Rename and format
    if "labels" not in dataset["train"].column_names:
        dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(config.LABELS)
    )

    training_args = TrainingArguments(
        output_dir=config.MODEL_PATH,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        logging_steps=20,
        weight_decay=0.01,
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # ✅ Evaluate on test set
    metrics = trainer.evaluate(test_dataset)
    print("Test Set Evaluation:", metrics)

    # ✅ Save model, tokenizer, and labels
    model.save_pretrained(config.MODEL_PATH)
    tokenizer.save_pretrained(config.MODEL_PATH)
    with open(f"{config.MODEL_PATH}/labels.json", "w") as f:
        json.dump(config.LABELS, f)



def predict_emotion(text):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_PATH)

    # Handle device (optional GPU support)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_label = torch.argmax(probs, dim=1).item()
    confidence = torch.max(probs).item()

    # Load labels if not already set
    if not config.LABELS:
        labels_path = os.path.join(config.MODEL_PATH, "labels.json")
        with open(labels_path, "r") as f:
            config.LABELS = json.load(f)

    return config.LABELS[predicted_label], confidence
