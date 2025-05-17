# Manufacturing-Work-Order-Generator---Knowledge-Injection-and-Distillation

- This model predicts the severity level (Low, Medium, High) of a maintenance issue based on its description and location.  
- It fine-tunes DistilBERT using LoRA to classify issues efficiently with fewer trainable parameters.
- The output helps prioritize maintenance tasks by assessing the urgency of different failures in the manufacturing process.
- Given an issue it can predict the severity and allocate a solution. 

1️. data.py → Generates synthetic maintenance logs and saves them to maintenance_logs.csv.

2️. train_teacher.py → Trains the large teacher model (bert-base-uncased).

3️. train_lora.py → Fine-tunes the LoRA-adapted smaller model (distilbert-base-uncased).

4️. distill_model.py → Transfers knowledge from the teacher to the student model.

5️. inference.py → Runs inference on new maintenance issues to predict severity and generate work orders.

LoRA - Low Rank Adapter : 

![image](https://github.com/user-attachments/assets/5479e46d-0c09-4f1d-a69a-113a692d5a44)

Knowledge Injection : 

![image](https://github.com/user-attachments/assets/e5030935-9881-477d-84d9-fe61817e2123)


How does this Mainteance Work Order Generator work 
Step 1: Generate Synthetic Data (data.py) - Creates a CSV file (maintenance_logs.csv) containing sample maintenance issues.

Step 2: Train the Teacher Model (train_teacher.py) - Trains BERT (teacher model) on the dataset.

Step 3: Train the LoRA Model (train_lora.py) - Fine-tunes DistilBERT with LoRA adapters for efficiency.

Step 4: Knowledge Distillation (distill_model.py) - Transfers knowledge from BERT (teacher) to DistilBERT (student) to make it smaller and faster.

Step 5: Run Inference (inference.py) - Uses the trained model to classify issue severity and recommend actions.

Results from Injection Training : 

![lora_training_progress](https://github.com/user-attachments/assets/cb12196c-4e1e-49b8-a7f8-fb1f11ce258c)

Distillation Training : 

![training_progress](https://github.com/user-attachments/assets/33faec53-0481-4eab-bd0a-f214925ff8ae)

Inference Results : 

![image](https://github.com/user-attachments/assets/8dfa4733-4118-4571-bd07-3b1a7b66a914)


