# XAI Framework for Bias Detection and Mitigation in Healthcare AI

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg" alt="PyTorch Version"/>
  <img src="https://img.shields.io/badge/status-complete-brightgreen.svg" alt="Project Status"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/>
</p>

---

## 1. Introduction: The Problem of Trust in Medical AI

Artificial Intelligence holds immense promise for revolutionizing healthcare diagnostics. However, a critical barrier to its adoption is the "black box" problem and the risk of hidden biases. An AI model might achieve high accuracy in a lab setting, but what if it does so for the wrong reasons? What if it learns to associate diagnoses with demographic data like ethnicity or gender instead of relying purely on medical evidence? Such a model would be untrustworthy and potentially dangerous.

This project tackles this challenge head-on. It presents a complete, end-to-end framework to **detect, analyze, and actively mitigate** demographic bias in a sophisticated multimodal diagnostic AI. We don't just build a model; we build a system of accountability around it.

Our approach uses **Explainable AI (XAI)**, specifically the SHAP library, as a diagnostic tool for the model itself. By understanding *why* our model makes a decision, we can identify if it's "cheating" by using biased shortcuts. We then demonstrate how to train a new, fairer model and use the same XAI tools to **prove** that the new model is more robust and trustworthy.

## 2. Our Journey: Thinking Process and Project Progression

This project was developed iteratively, following a clear plan to solve a complex problem.

1.  **The Initial Idea & Four-Phase Plan:** We started with a clear goal: create a framework to find and fix bias. We structured this into four logical phases: (1) Data Simulation, (2) Baseline Model Training, (3) XAI Analysis, and (4) Bias Mitigation & Re-evaluation.

2.  **The Power of Synthetic Data:** We made a crucial decision to use a synthetic dataset. While real-world data is the ultimate test, a synthetic approach gives us a controlled "laboratory." It allowed us to inject a **specific, known bias** (correlating a patient's "group" with their diagnosis) and then test if our framework could correctly identify and remove this exact bias. This is the scientific method applied to model development.

3.  **Overcoming Technical Hurdles:** Development wasn't a straight line. We encountered and solved several real-world engineering challenges:
    *   **SHAP Dimensionality:** The SHAP library had strict requirements for the shape of image data. We engineered a solution by creating wrapper functions to flatten and reshape the data on the fly, allowing the tool to work with our complex model.
    *   **Robust Pathing:** We initially had a classic relative-path issue where output files were being saved outside the project directory. We corrected this by refactoring all scripts to use absolute paths derived from their file location, making the project self-contained and portable.

4.  **The "Smoking Gun" Discovery:** Our most significant insight came during the analysis phase. Our initial hypothesis was that we'd see a subtle change in the model's feature importances. The actual result was far more dramatic: the biased baseline model **completely ignored the clinical text** for patients from the biased demographic group. This was the "smoking gun" proving it was relying on the demographic shortcut. This discovery made the success of the mitigation even more evident and impactful.

## 3. How the Framework Works: A Deep-Dive into Each Component

Each file in the `src/` directory plays a critical role in the pipeline.

-   `src/data_simulator.py`
    -   **Contribution:** Creates our controlled experimental dataset.
    -   **Details:** It generates a `metadata.csv` file and a corresponding folder of images. It creates two patient groups, "A" and "B". The `bias_factor` parameter is used to make a positive diagnosis much more likely for patients in "Group B," injecting the bias we aim to detect.

-   `src/dataset.py`
    -   **Contribution:** Prepares the data for PyTorch.
    -   **Details:** This script defines a standard PyTorch `Dataset` class. It handles loading images, applying transformations (resizing, normalizing), and tokenizing the text notes. It also builds a `Vocabulary` to convert words into numerical tokens that the model can understand.

-   `src/model.py`
    -   **Contribution:** Defines the AI's "brain."
    -   **Details:** This script contains our `MultimodalModel`. It has two branches: (1) a **vision branch** using a pre-trained ResNet18 to analyze images, and (2) a **text branch** using an LSTM network to analyze the sequence of words in the clinical notes. The features from these two branches are then fused in a final classification head to make a single diagnosis.

-   `src/train.py`
    -   **Contribution:** Trains the initial, biased baseline model.
    -   **Details:** This is a standard PyTorch training loop that uses the dataset and model defined above. It trains the model to achieve the highest possible accuracy on the data and saves the result as `models/best_model.pth`.

-   `src/mitigate.py`
    -   **Contribution:** Trains the new, fairer model.
    -   **Details:** This is the core of our intervention. It uses `sklearn.utils.class_weight.compute_sample_weight` to calculate weights that give more importance to under-represented samples. The PyTorch loss function (`BCEWithLogitsLoss`) is then applied on a per-sample basis, multiplied by these weights. This discourages the model from relying on the easy, biased patterns and saves the improved model as `models/best_model_mitigated.pth`.

-   `src/explain.py`
    -   **Contribution:** The diagnostic tool that lets us look inside the models.
    -   **Details:** This is the most complex script. It loads a saved model and uses the SHAP library to generate explanations. Because SHAP's `KernelExplainer` can only handle one input at a time, we created two separate wrapper functions (`predict_for_image` and `predict_for_text`) that "trick" SHAP into analyzing one modality while holding the other constant. It then generates and saves the visual plots that are the ultimate output of our analysis.

## 4. Running the Experiment: A Step-by-Step Guide

Follow these steps to reproduce the entire experiment from scratch.

*(The Setup and Installation instructions from the previous README would be here)*

### Step 1: Generate the Dataset
This script creates the `data/` directory, populates it with synthetic images, and creates the `metadata.csv` file.
`python src/data_simulator.py`

### Step 2: Train the Baseline Model
This trains the initial model on the biased dataset and saves the best version to `models/best_model.pth`.
`python src/train.py`

### Step 3: Analyze the Baseline Model for Bias
This script loads the baseline model and generates SHAP explanation plots. Before running, ensure it's configured to analyze the baseline model (`MODEL_PATH = ".../best_model.pth"`) and save plots without a suffix.
`python src/explain.py`

### Step 4: Train the Bias-Mitigated Model
This trains a new model using our fairness-enhancing technique.
`python src/mitigate.py`

### Step 5: Analyze the Mitigated Model
Modify `src/explain.py` to point to the new model (`MODEL_PATH = ".../best_model_mitigated.pth"`) and change the output filenames to include a `_mitigated` suffix. Then, run the script again.
`python src/explain.py`

## 5. Analyzing the Results: A Guided Walkthrough

The primary result of this project is the **comparison** between the baseline and mitigated models, made visible by the SHAP explanation plots.

#### The Baseline Model: Accurate but Biased
- The initial model (`best_model.pth`) quickly achieves a perfect accuracy score.
- However, the SHAP analysis reveals a significant flaw. For patients in the over-represented "Group B," the model may learn to ignore the text-based clinical evidence entirely.
- **The "Smoking Gun":** When you view the text explanation plot (e.g., `PID_00008_text_explanation.png`), you may see **no significant bars**. This indicates that the model was so confident based on the patient's demographic group that it considered the medical notes irrelevant to its decision. This is a dangerous form of automation bias.

#### The Mitigated Model: Accurate and Fair
- The mitigated model (`best_model_mitigated.pth`) is trained to disregard the demographic shortcut by using a weighted loss function.
- While it also achieves perfect accuracy (due to the clear signals in the synthetic data), its reasoning process is fundamentally different and more robust.
- **Verification of Success:** When you view the mitigated text explanation plot (e.g., `PID_00008_text_explanation_mitigated.png`), you will now see **long, red bars** next to diagnostic keywords like "abnormal" and "high-risk." This proves the model is now basing its decision on the actual medical evidence in the notes.

By comparing the two sets of plots, you can visually confirm that the mitigation strategy was successful. We developed a model that is not only accurate but also more trustworthy and fair, as it relies on the correct features for its diagnosis.

## 6. Future Directions

This framework serves as a strong foundation. Future enhancements could include:
-   **Advanced Mitigation Techniques**: Implementing other methods like adversarial debiasing or projection-based techniques.
-   **Real-World Datasets**: Adapting the pipeline to work with well-known public healthcare datasets (e.g., MIMIC-CXR, CheXpert).
-   **Interactive Visualization**: Building a simple web interface (e.g., with Streamlit or Flask) to interactively explore model explanations.

## 7. License

This project is licensed under the MIT License.
