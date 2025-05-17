import pandas as pd
from faker import Faker
import random

fake = Faker()

# Define high-level issue categories and specific issues within them
issue_categories = {
    "Assembly Line": ["Conveyor belt misalignment", "Robotic arm calibration failure", "Paint sprayer clogging", "Spot welding machine misfire"],
    "Parts & Supply": ["Delayed shipment of brake pads", "Defective windshield batch", "Engine block casting defects", "Incorrect tire specifications received"],
    "Machinery & Equipment": ["CNC machine overheating", "Hydraulic press pressure loss", "Lathe machine spindle failure", "Injection molding machine nozzle clog"],
    "Sensor & Electrical": ["Faulty RFID scanner at inventory check", "PLC (Programmable Logic Controller) failure", "Battery pack overheating in EV assembly", "Electrical surge causing sensor malfunctions"],
    "Logistics & Storage": ["Tire storage humidity too high", "Forklift fuel leakage", "Barcode scanner failure in warehouse", "Inventory mismatch in parts database"]
}

# Define severity rules based on issues
severity_rules = {
    "overheating": "High",
    "failure": "High",
    "defective": "Medium",
    "misalignment": "Medium",
    "clogging": "Medium",
    "delayed": "Low",
    "incorrect": "Low",
    "mismatch": "Low"
}

def determine_severity(issue):
    """Determine severity based on keywords in the issue description"""
    for keyword, severity in severity_rules.items():
        if keyword in issue.lower():
            return severity
    return random.choice(["Low", "Medium", "High"])

# Generate synthetic maintenance logs with more balanced data
data = []
num_samples = 3000  # Increased to 3000 samples

for _ in range(num_samples):
    timestamp = fake.date_time_this_year()
    machine_id = f"M{random.randint(100, 999)}"
    
    category = random.choice(list(issue_categories.keys()))
    issue_description = random.choice(issue_categories[category])
    
    # Determine severity based on issue description
    severity = determine_severity(issue_description)
    location = fake.city()

    data.append([timestamp, machine_id, category, issue_description, severity, location])

# Create DataFrame
df = pd.DataFrame(data, columns=["timestamp", "machine_id", "category", "issue_description", "severity", "location"])

# Check class distribution
print("\nClass Distribution:")
print(df['severity'].value_counts(normalize=True))

# Save to CSV
df.to_csv("maintenance_logs.csv", index=False)
print("\nMaintenance logs generated and saved successfully!")
print(f"Total samples: {len(df)}")


'''
1️⃣ data.py → Generates synthetic maintenance logs and saves them to maintenance_logs.csv.
2️⃣ train_teacher.py → Trains the large teacher model (bert-base-uncased).
3️⃣ train_lora.py → Fine-tunes the LoRA-adapted smaller model (distilbert-base-uncased).
4️⃣ distill_model.py → Transfers knowledge from the teacher to the student model.
5️⃣ inference.py → Runs inference on new maintenance issues to predict severity and generate work orders.

'''

'''
🔹 Step 1: Generate Synthetic Data (data.py)

Creates a CSV file (maintenance_logs.csv) containing sample maintenance issues.
🔹 Step 2: Train the Teacher Model (train_teacher.py)

Trains BERT (teacher model) on the dataset.
🔹 Step 3: Train the LoRA Model (train_lora.py)

Fine-tunes DistilBERT with LoRA adapters for efficiency.
🔹 Step 4: Knowledge Distillation (distill_model.py)

Transfers knowledge from BERT (teacher) to DistilBERT (student) to make it smaller and faster.
🔹 Step 5: Run Inference (inference.py)

Uses the trained model to classify issue severity and recommend actions.
'''

''' TERMINAL OUTPUT
Class Distribution:
severity
High      0.423000
Low       0.292667
Medium    0.284333
Name: proportion, dtype: float64

Maintenance logs generated and saved successfully!
Total samples: 3000

'''