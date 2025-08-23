
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# Get the absolute path to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

class HealthcareDataSimulator:
    """
    Generates a synthetic multimodal healthcare dataset (images + text)
    with controllable, built-in biases for research and model development.

    The simulation creates a scenario where a specific demographic group has a
    higher prevalence of a certain diagnosis, but this correlation is also
    artificially amplified, creating a bias that a model might exploit.
    """

    def __init__(self, num_samples, bias_factor=0.8):
        """
        Initializes the simulator.

        Args:
            num_samples (int): The total number of patient samples to generate.
            bias_factor (float): The strength of the bias. 0.5 means no bias,
                                 1.0 means perfect correlation between the biased
                                 group and the diagnosis.
        """
        self.num_samples = num_samples
        # Correctly define output paths relative to the project root
        self.output_dir = os.path.join(PROJECT_ROOT, 'data')
        self.image_dir = os.path.join(self.output_dir, "images")
        self.bias_factor = bias_factor
        
        # Ensure output directories exist
        os.makedirs(self.image_dir, exist_ok=True)

        # Define keywords for clinical notes based on diagnosis
        self.positive_keywords = ["abnormal cells", "lesion detected", "high-risk", "positive result", "mass found", "irregular shape"]
        self.negative_keywords = ["normal", "clear", "no abnormalities", "benign", "negative result", "routine check-up"]

    def _generate_synthetic_image(self, diagnosis, image_path):
        """
        Generates and saves a synthetic medical image.

        - Positive diagnosis: A dark gray image with a small, bright white square (lesion).
        - Negative diagnosis: A dark gray image with random noise.
        """
        img_size = (128, 128)
        bg_color = (50, 50, 50)
        image = Image.new("RGB", img_size, bg_color)
        draw = ImageDraw.Draw(image)

        if diagnosis == 1:
            # Draw a "lesion" for positive diagnosis
            lesion_size = 20
            x1 = np.random.randint(20, img_size[0] - lesion_size - 20)
            y1 = np.random.randint(20, img_size[1] - lesion_size - 20)
            x2 = x1 + lesion_size
            y2 = y1 + lesion_size
            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
        else:
            # Add random noise for negative diagnosis
            for _ in range(100):
                x = np.random.randint(0, img_size[0])
                y = np.random.randint(0, img_size[1])
                noise_color = tuple(np.random.randint(0, 70, 3))
                draw.point((x, y), fill=noise_color)

        image.save(image_path)

    def _generate_clinical_note(self, diagnosis):
        """
        Generates a short clinical note based on the diagnosis.
        """
        if diagnosis == 1:
            base_text = "Patient presents with symptoms. Examination reveals "
            keywords = np.random.choice(self.positive_keywords, 2, replace=False)
            return base_text + ", ".join(keywords) + "."
        else:
            base_text = "Patient is asymptomatic. Examination is "
            keywords = np.random.choice(self.negative_keywords, 2, replace=False)
            return base_text + ", ".join(keywords) + "."

    def generate_dataset(self):
        """
        The main method to generate the full dataset and save it.
        
        This method orchestrates the generation of demographics, diagnoses (with bias),
        images, and clinical notes, and saves everything to a CSV file.
        """
        records = []
        
        # 1. Generate Demographics
        demographics = {
            "patient_id": [f"PID_{i:05d}" for i in range(self.num_samples)],
            "age": np.random.randint(20, 80, self.num_samples),
            "sex": np.random.choice(["Male", "Female"], self.num_samples, p=[0.5, 0.5]),
            # This is our sensitive attribute for introducing bias
            "group": np.random.choice(["A", "B"], self.num_samples, p=[0.7, 0.3])
        }
        df = pd.DataFrame(demographics)

        # 2. Generate Diagnosis with BIAS
        # We will create a correlation between 'group' and 'diagnosis'.
        # Group 'B' will have a much higher chance of a positive diagnosis.
        
        diagnoses = []
        for _, row in df.iterrows():
            is_group_b = (row["group"] == "B")
            # The probability of positive diagnosis is much higher for group B
            prob_positive = self.bias_factor if is_group_b else (1 - self.bias_factor) / 2
            
            diagnosis = np.random.choice([1, 0], p=[prob_positive, 1 - prob_positive])
            diagnoses.append(diagnosis)
            
        df["diagnosis"] = diagnoses

        # 3. Generate Images and Notes based on the *actual* diagnosis
        print("Generating images and clinical notes...")
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Samples"):
            patient_id = row["patient_id"]
            diagnosis = row["diagnosis"]
            
            # Generate and save image
            image_filename = f"{patient_id}.png"
            image_path = os.path.join(self.image_dir, image_filename)
            self._generate_synthetic_image(diagnosis, image_path)
            
            # Generate clinical note
            note = self._generate_clinical_note(diagnosis)
            
            records.append({
                "patient_id": patient_id,
                "age": row["age"],
                "sex": row["sex"],
                "group": row["group"],
                "image_path": os.path.join("images", image_filename), # Relative path for the CSV
                "clinical_note": note,
                "diagnosis": diagnosis
            })
            
        final_df = pd.DataFrame(records)
        metadata_path = os.path.join(self.output_dir, "metadata.csv")
        final_df.to_csv(metadata_path, index=False)
        
        print(f"\nDataset generation complete!")
        print(f"Generated {self.num_samples} samples.")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Images saved in: {self.image_dir}")
        
        # Print bias summary
        print("\n--- Bias Analysis ---")
        print(pd.crosstab(final_df['group'], final_df['diagnosis']))
        print("---------------------")


if __name__ == '__main__':
    # Example of how to use the simulator
    simulator = HealthcareDataSimulator(num_samples=1000, bias_factor=0.8)
    simulator.generate_dataset()
