# Train the teacher model - Train BERT model on maintenance logs
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from typing import List, Dict

# Load maintenance logs
df = pd.read_csv("maintenance_logs.csv")

# Define severity levels as labels
severity_map = {"Low": 0, "Medium": 1, "High": 2}
df["severity_label"] = df["severity"].map(severity_map)

# Define a dataset class
class MaintenanceDataset(Dataset):
    def __init__(self, tokenizer, dataframe):
        self.tokenizer = tokenizer
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = f"{row['category']} - {row['issue_description']} in {row['location']}"
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        label = torch.tensor(row["severity_label"], dtype=torch.long)
        return {**{k: v.squeeze(0) for k, v in encoding.items()}, "labels": label}

# Load BERT model & tokenizer
model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Prepare dataset
dataset = MaintenanceDataset(tokenizer, df)

# Training setup
training_args = TrainingArguments(
    output_dir="./bert_teacher_model",
    per_device_train_batch_size=8,
    num_train_epochs=20,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no"
)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_loss = []
        self.epoch_losses: List[float] = []
        self.current_epoch_loss = 0.0
        self.steps_in_epoch = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to track loss."""
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.training_loss.append(loss.item())
        self.current_epoch_loss += loss.item()
        self.steps_in_epoch += 1
        return loss

    def on_epoch_end(self):
        if self.steps_in_epoch > 0:  # Add check to prevent division by zero
            avg_epoch_loss = self.current_epoch_loss / self.steps_in_epoch
            self.epoch_losses.append(avg_epoch_loss)
            self.current_epoch_loss = 0.0
            self.steps_in_epoch = 0
            
            # Plot progress after each epoch
            self.plot_training_progress()

    def plot_training_progress(self):
        plt.figure(figsize=(10, 6))
        
        # Plot per-step loss
        plt.subplot(1, 2, 1)
        plt.plot(self.training_loss, label='Training Loss')
        plt.title('Loss per Step')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot per-epoch loss
        plt.subplot(1, 2, 2)
        plt.plot(self.epoch_losses, 'r-o', label='Epoch Loss')
        plt.title('Average Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

# Replace Trainer with CustomTrainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train teacher model
trainer.train()
model.save_pretrained("bert_teacher_model")

# Final plot
trainer.plot_training_progress()
print("Teacher model saved!")
print("Training visualization saved as 'training_progress.png'")
