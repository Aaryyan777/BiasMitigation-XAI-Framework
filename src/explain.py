
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import json
import os

from model import MultimodalModel
from dataset import HealthcareMultimodalDataset, Vocabulary # We need the vocab

# --- Configuration ---
# Define project root and paths relative to it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_FILE = os.path.join(DATA_DIR, "metadata.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model_mitigated.pth")
NUM_SAMPLES_TO_EXPLAIN = 5

# 1. Load model and dataset
print("Loading model and dataset...")
device = torch.device("cpu") # SHAP runs on CPU

# Load the dataset to get the vocabulary
full_dataset = HealthcareMultimodalDataset(csv_file=CSV_FILE, root_dir=DATA_DIR)
vocab = full_dataset.vocab
vocab_size = len(vocab)

# Instantiate and load the trained model
model = MultimodalModel(vocab_size=vocab_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Model and dataset loaded successfully.")

# 2. Create a wrapper prediction function for SHAP
# SHAP's KernelExplainer expects a function that takes a numpy array and returns a numpy array.
# Our model takes two tensors (image, text). We need to create a wrapper.

def model_prediction_wrapper(image_np, text_np):
    """
    A wrapper to handle preprocessing and prediction for a batch of samples.
    """
    # Preprocess images
    image_tensors = []
    for img_arr in image_np:
        img = Image.fromarray(img_arr.astype(np.uint8))
        img_tensor = full_dataset.transform(img)
        image_tensors.append(img_tensor)
    image_batch = torch.stack(image_tensors).to(device)

    # Preprocess texts
    text_tensors = []
    for text_arr in text_np:
        # The text comes in as an array of numbers, we need to convert it back to a string
        note = " ".join([full_dataset.vocab.idx2word[int(i)] for i in text_arr if i != 0])
        tokens = full_dataset.tokenize_and_pad(note)
        text_tensors.append(torch.LongTensor(tokens))
    text_batch = torch.stack(text_tensors).to(device)

    # Get model output
    with torch.no_grad():
        logits = model(image_batch, text_batch)
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy()

# We need two separate prediction functions for SHAP, one for each modality
def predict_for_image(image_np_flattened):
    """
    Prediction function for explaining images. Text is held constant.
    Accepts a flattened numpy array of images as required by KernelExplainer.
    """
    # Reshape the flattened images back to their original shape
    image_np = image_np_flattened.reshape(-1, 224, 224, 3)
    
    # Use a fixed, neutral text input as the background
    neutral_text_tokens = full_dataset.tokenize_and_pad("patient check-up")
    text_np = np.array([neutral_text_tokens] * image_np.shape[0])
    return model_prediction_wrapper(image_np, text_np)

def predict_for_text(text_np):
    """Prediction function for explaining text. Image is held constant."""
    # Use a fixed, neutral image input as the background (e.g., a gray image)
    neutral_image = np.full((text_np.shape[0], 224, 224, 3), 128)
    return model_prediction_wrapper(neutral_image, text_np)


# 3. Select data for explanation and create explainers
# We use a subset of the training data as the background for the explainer
background_data_indices = np.random.choice(len(full_dataset), 50, replace=False)

# Prepare image background data and flatten it for the explainer
image_background = []
for i in background_data_indices:
    img_path = os.path.join(DATA_DIR, full_dataset.metadata.iloc[i, 4])
    img = Image.open(img_path).resize((224, 224))
    image_background.append(np.array(img))
image_background = np.array(image_background).reshape(len(background_data_indices), -1)


# Prepare text background data
text_background = []
for i in background_data_indices:
    note = full_dataset.metadata.iloc[i, 5]
    tokens = full_dataset.tokenize_and_pad(note)
    text_background.append(tokens)
text_background = np.array(text_background)

# Create SHAP Explainers
print("\nCreating SHAP explainers...")
image_explainer = shap.KernelExplainer(predict_for_image, image_background)
text_explainer = shap.KernelExplainer(predict_for_text, text_background)

# 4. Generate and plot explanations for a few samples
print(f"Generating explanations for {NUM_SAMPLES_TO_EXPLAIN} samples...")

# Select some interesting samples to explain (e.g., from the biased group B)
samples_to_explain_df = full_dataset.metadata[full_dataset.metadata['group'] == 'B'].head(NUM_SAMPLES_TO_EXPLAIN)

for index, row in samples_to_explain_df.iterrows():
    print(f"\n--- Explaining Sample: {row['patient_id']} (Group: {row['group']}, Diagnosis: {row['diagnosis']}) ---")
    
    # Get the data for the sample
    img_path = os.path.join(DATA_DIR, row['image_path'])
    original_image = Image.open(img_path).resize((224, 224))
    image_to_explain_orig = np.array([np.array(original_image)])
    image_to_explain_flat = image_to_explain_orig.reshape(1, -1)
    
    note = row['clinical_note']
    tokens = full_dataset.tokenize_and_pad(note)
    text_to_explain = np.array([tokens])
    
    # --- Image Explanation ---
    shap_values_image_flat = image_explainer.shap_values(image_to_explain_flat, nsamples=100)
    
    # --- Text Explanation ---
    shap_values_text = text_explainer.shap_values(text_to_explain, nsamples=100)
    
    # For text, we need to map shap values back to words
    words = [vocab.idx2word[i] for i in tokens if i != 0] # 0 is pad
    shap_values_text_squeezed = shap_values_text[0].flatten()
    
    print("Plotting explanations...")
    # Reshape image shap values for plotting
    shap_values_image = shap_values_image_flat.reshape(image_to_explain_orig.shape)

    # Plot image explanation
    shap.image_plot(shap_values_image, image_to_explain_orig, show=False)
    plt.suptitle(f"Image Explanation for {row['patient_id']}")
    plt.savefig(os.path.join(DATA_DIR, f"{row['patient_id']}_image_explanation_mitigated.png"))
    plt.close()


    # Create and save text explanation plot
    # A custom plot is often better for text
    fig, ax = plt.subplots()
    y_pos = np.arange(len(words))
    ax.barh(y_pos, shap_values_text_squeezed[:len(words)], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("SHAP Value")
    ax.set_title(f"Text Explanation for {row['patient_id']}")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, f"{row['patient_id']}_text_explanation_mitigated.png"))
    plt.close()

    print(f"Saved explanation plots for {row['patient_id']}.")

print("\nExplanation generation complete.")
