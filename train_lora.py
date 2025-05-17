#Fine-tune the distilled model using LoRA for efficiency 
import pandas as pd
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# Load base model (DistilBERT for classification)
model_name = "distilbert-base-uncased"
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 labels: Low, Medium, High
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Apply LoRA adapter - LoRA Adapters inject new knowledge into specific parts of the model - attention layers while freezing most of the model
lora_config = LoraConfig(
    r=16,  # Increase rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin", "k_lin", "out_lin"],  # Target more attention components
    bias="none",
    task_type="SEQ_CLS"  # Specify task type
)
# We preserve distialled knowledge from BERRT and we inject new domain specific knowledge from maintenance logs 
model = get_peft_model(model, lora_config)

# Load maintenance logs
df = pd.read_csv("maintenance_logs.csv")

# Convert severity to labels
severity_map = {"Low": 0, "Medium": 1, "High": 2}
df["severity_label"] = df["severity"].map(severity_map)

# Check class distribution in your data
print("Label distribution:")
print(df['severity_label'].value_counts(normalize=True))

# Define dataset class
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

# Prepare dataset and dataloader
dataset = MaintenanceDataset(tokenizer, df)

# After dataset creation and before data loaders, add hyperparameters
# Training hyperparameters
initial_lr = 1e-4  # Increase learning rate
batch_size = 32    # Adjust batch size
num_epochs = 50

# Add gradient accumulation
gradient_accumulation_steps = 4

# Split dataset into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Add this after creating the datasets
print("\nTraining set size:", len(train_dataset))
print("Validation set size:", len(val_dataset))

# Create data loaders (now batch_size is defined)
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
)

# Recalculate total steps based on train loader length
total_steps = len(train_loader) * num_epochs
warmup_steps = int(total_steps * 0.1)

# Training setup with improvements
optimizer = AdamW(model.parameters(), lr=initial_lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

def evaluate(model, val_loader, device):
    """Run validation."""
    model.eval()
    val_losses = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_losses.append(outputs.loss.item())
            
            # Calculate accuracy
            predictions = outputs.logits.argmax(-1)
            correct += (predictions == batch['labels']).sum().item()
            total += len(batch['labels'])
    
    avg_val_loss = np.mean(val_losses)
    accuracy = correct / total
    return avg_val_loss, accuracy

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Training loop with progress bars and validation
best_val_loss = float('inf')
train_losses = []
val_losses = []
accuracies = []

early_stopping = EarlyStopping(patience=5)

for epoch in trange(num_epochs, desc="Epochs"):
    model.train()
    batch_losses = []
    
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps  # Scale loss
        
        loss.backward()
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        batch_losses.append(loss.item() * gradient_accumulation_steps)  # Unscale loss for logging
        
        # Add more detailed logging
        if batch_idx % 10 == 0:
            predictions = outputs.logits.argmax(-1)
            batch_accuracy = (predictions == batch['labels']).float().mean()
            progress_bar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'batch_acc': f'{batch_accuracy:.2%}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
    
    # Calculate training loss
    avg_train_loss = np.mean(batch_losses)
    train_losses.append(avg_train_loss)
    
    # Validation
    val_loss, accuracy = evaluate(model, val_loader, model.device)
    val_losses.append(val_loss)
    accuracies.append(accuracy)
    
    # Print epoch summary
    print(f"\nEpoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Accuracy: {accuracy:.2%}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained("lora_work_order_model_best")
        print("  Saved new best model!")

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# Plot training progress
plt.figure(figsize=(15, 5))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Validation Accuracy', color='green')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_progress.png')
plt.close()

print("\nTraining completed!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Final validation accuracy: {accuracies[-1]:.2%}")

# Print model architecture to verify LoRA application
print("Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

