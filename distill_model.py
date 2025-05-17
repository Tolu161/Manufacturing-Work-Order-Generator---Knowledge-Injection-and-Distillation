# Distill the teacher model into a smaller student model , Distill BERT into DistilBERT 

import torch
from transformers import DistilBertForSequenceClassification, BertForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np

# Load models
teacher_model = BertForSequenceClassification.from_pretrained("bert_teacher_model")
student_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Load tokenizers
teacher_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
student_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load data
df = pd.read_csv("maintenance_logs.csv")
severity_map = {"Low": 0, "Medium": 1, "High": 2}
df["severity_label"] = df["severity"].map(severity_map)

# Dataset class
class MaintenanceDataset(torch.utils.data.Dataset):
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

# Prepare datasets and dataloaders
train_dataset = MaintenanceDataset(student_tokenizer, df)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
student_model.to(device)

teacher_model.eval()  # Teacher in eval mode
student_model.train()  # Student in training mode

# Distillation parameters
temperature = 2.0
alpha = 0.5  # Weight for distillation loss
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
num_epochs = 10

# Distillation loss function
def distillation_loss(student_logits, teacher_logits, labels, temp, alpha):
    """Compute the distillation loss."""
    distill_loss = torch.nn.KLDivLoss(reduction='batchmean')(
        torch.nn.functional.log_softmax(student_logits / temp, dim=1),
        torch.nn.functional.softmax(teacher_logits / temp, dim=1)
    ) * (temp * temp)
    
    student_loss = torch.nn.CrossEntropyLoss()(student_logits, labels)
    return alpha * distill_loss + (1 - alpha) * student_loss

# Training loop
print("Starting knowledge distillation...")
for epoch in range(num_epochs):
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_outputs = teacher_model(**batch)
            teacher_logits = teacher_outputs.logits
        
        # Get student predictions
        student_outputs = student_model(**batch)
        student_logits = student_outputs.logits
        
        # Calculate loss
        loss = distillation_loss(
            student_logits,
            teacher_logits,
            batch['labels'],
            temperature,
            alpha
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

# Save distilled model
student_model.save_pretrained("distilled_work_order_model")
print("Distilled model saved successfully!")
