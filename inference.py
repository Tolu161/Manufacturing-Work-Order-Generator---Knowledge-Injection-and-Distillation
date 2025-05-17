import torch
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import numpy as np

class MaintenancePredictor:
    def __init__(self, model_path="distilled_work_order_model"):
        # Load the distilled model and tokenizer
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Define severity levels and actions
        self.severity_labels = {0: "Low", 1: "Medium", 2: "High"}
        self.recommended_actions = {
            "Low": [
                "Schedule routine maintenance within the next week.",
                "Document in maintenance log for next inspection.",
                "Monitor for any changes in condition."
            ],
            "Medium": [
                "Assign technician for inspection within 24 hours.",
                "Prepare necessary replacement parts.",
                "Schedule maintenance window with production team."
            ],
            "High": [
                "Immediate technician dispatch required!",
                "Alert shift supervisor and maintenance lead.",
                "Initiate emergency response protocol if safety-critical."
            ]
        }

    def predict(self, issues, batch_size=8):
        """Predict severity and recommend actions for multiple issues."""
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(issues), batch_size), desc="Processing issues"):
            batch_issues = issues[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_issues,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

            # Process each prediction in the batch
            for issue, pred, prob in zip(batch_issues, predictions, probs):
                severity = self.severity_labels[pred.item()]
                confidence = prob[pred.item()].item()
                actions = self.recommended_actions[severity]
                
                results.append({
                    "issue": issue,
                    "severity": severity,
                    "confidence": f"{confidence:.2%}",
                    "actions": actions,
                    "probabilities": {
                        label: f"{p:.2%}" for label, p in 
                        zip(self.severity_labels.values(), prob.cpu().numpy())
                    }
                })

        return results

    def generate_report(self, results):
        """Generate a formatted report of predictions."""
        print("\n=== Maintenance Issue Analysis Report ===\n")
        
        for i, result in enumerate(results, 1):
            print(f"Issue {i}: {result['issue']}")
            print(f"Severity: {result['severity']} (Confidence: {result['confidence']})")
            print("\nProbability Distribution:")
            for severity, prob in result['probabilities'].items():
                print(f"  {severity}: {prob}")
            print("\nRecommended Actions:")
            for action in result['actions']:
                print(f"  • {action}")
            print("\n" + "="*50 + "\n")

def main():
    # Initialize predictor
    predictor = MaintenancePredictor()

    # Example test cases
    test_issues = [
        "Motor overheating in Factory Floor 3",
        "Sensor malfunction detected in Warehouse 2",
        "Routine maintenance required for conveyor belt",
        "Hydraulic pressure drop in Assembly Line 1",
        "Delayed shipment of brake pads affecting production",
        "Defective windshield batch found in stock",
        "Critical failure in robotic welding arm",
        "Minor oil leak in maintenance bay",
        "Emergency stop triggered on production line"
    ]

    # Get predictions
    results = predictor.predict(test_issues)
    
    # Generate report
    predictor.generate_report(results)

if __name__ == "__main__":
    main()



