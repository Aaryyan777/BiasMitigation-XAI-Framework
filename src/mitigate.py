
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
import numpy as np

# Import our custom modules
from model import MultimodalModel
from dataset import HealthcareMultimodalDataset


# Define project root and paths relative to it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_FILE = os.path.join(DATA_DIR, "metadata.csv")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "best_model_mitigated.pth")

# Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42

def train_mitigated_model():
    """Main function to orchestrate the bias-mitigated model training and validation process."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset
    print("Loading dataset...")
    full_dataset = HealthcareMultimodalDataset(csv_file=CSV_FILE, root_dir=DATA_DIR)
    vocab_size = len(full_dataset.vocab)

    # 2. Split Dataset
    dataset_indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        dataset_indices, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=full_dataset.metadata.diagnosis
    )
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Training set size: {len(train_subset)}, Validation set size: {len(val_subset)}")

    # 3. Calculate Sample Weights for Bias Mitigation
    print("Calculating sample weights for bias mitigation...")
    # We wanted to balance the influence of the biased 'group' feature.
    # So we will compute weights based on the combined classes of (group, diagnosis).
    y_train = full_dataset.metadata.iloc[train_indices]
    
    # Create a composite class from group and diagnosis for weighting
    composite_class = y_train['group'] + "_" + y_train['diagnosis'].astype(str)
    
    sample_weights = compute_sample_weight(class_weight='balanced', y=composite_class)
    sample_weights = torch.from_numpy(sample_weights).double()

    # 4. Initialize Model, Loss, and Optimizer
    print("Initializing model...")
    model = MultimodalModel(vocab_size=vocab_size).to(device)
    # The key change: We will use the `weight` parameter in the loss function.
    # The reduction is set to 'none' so we can apply weights manually.
    criterion = nn.BCEWithLogitsLoss(reduction='none') 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1.0

    # 5. Training and Validation Loop
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # --- Training ---
        model.train()
        train_loss = 0.0
        for i, (images, texts, labels) in enumerate(tqdm(train_loader, desc="Training")):
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            
            # Get the weights for the current batch
            start = i * BATCH_SIZE
            end = start + len(labels)
            batch_weights = sample_weights[start:end].float().to(device)

            optimizer.zero_grad()
            outputs = model(images, texts)
            
            # Apply weights to the loss
            unweighted_loss = criterion(outputs, labels)
            weighted_loss = (unweighted_loss.squeeze() * batch_weights).mean()

            weighted_loss.backward()
            optimizer.step()
            train_loss += weighted_loss.item() * images.size(0)

        # --- Validation (remains unweighted) ---
        model.eval()
        val_loss_unweighted = 0.0
        all_preds = []
        all_labels = []
        unweighted_criterion = nn.BCEWithLogitsLoss()
        with torch.no_grad():
            for images, texts, labels in tqdm(val_loader, desc="Validating"):
                images, texts, labels = images.to(device), texts.to(device), labels.to(device)
                outputs = model(images, texts)
                loss = unweighted_criterion(outputs, labels)
                val_loss_unweighted += loss.item() * images.size(0)
                
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_train_loss = train_loss / len(train_subset)
        avg_val_loss = val_loss_unweighted / len(val_subset)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds)
        val_recall = recall_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f} | Val F1: {val_f1:.4f} | Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model improved! Saved to {MODEL_SAVE_PATH}")

    print("\nMitigated training complete!")
    print(f"Best validation F1-score: {best_val_f1:.4f}")

if __name__ == '__main__':
    train_mitigated_model()
