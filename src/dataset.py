
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import re
from collections import Counter

class HealthcareMultimodalDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and preprocessing the multimodal healthcare data.
    It handles image transformations and text tokenization.
    """

    def __init__(self, csv_file, root_dir, vocab=None, max_seq_length=50):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            vocab (Vocabulary, optional): A pre-built vocabulary object. If None, a new
                                        vocabulary will be built from the data.
            max_seq_length (int): The maximum length for padding text sequences.
        """
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.max_seq_length = max_seq_length

        if vocab is None:
            self.vocab = self.build_vocabulary(self.metadata['clinical_note'])
        else:
            self.vocab = vocab

        # Defining image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_name = os.path.join(self.root_dir, self.metadata.iloc[idx, 4])
        image = Image.open(img_name).convert("RGB")
        image = self.transform(image)

        # Process text
        note = self.metadata.iloc[idx, 5]
        tokens = self.tokenize_and_pad(note)
        text = torch.LongTensor(tokens)

        # Get label
        label = torch.FloatTensor([self.metadata.iloc[idx, 6]])

        return image, text, label

    def tokenize_text(self, text):
        """Simple tokenizer that converts text to lowercase and splits by non-alphanumeric characters."""
        return re.findall(r'\b\w+\b', text.lower())

    def build_vocabulary(self, all_notes):
        """Builds a vocabulary from a series of text notes."""
        word_counts = Counter()
        for note in all_notes:
            word_counts.update(self.tokenize_text(note))
        
        # Create a vocabulary object
        vocab = Vocabulary()
        # Add special tokens
        vocab.add_word('<pad>') # padding
        vocab.add_word('<unk>') # unknown word

        # Add all words from the dataset
        for word, count in word_counts.items():
            if count > 0: # Could use a threshold here if needed
                vocab.add_word(word)
        return vocab

    def tokenize_and_pad(self, note):
        """Tokenizes a single note and pads it to max_seq_length."""
        tokens = self.tokenize_text(note)
        token_indices = [self.vocab.word2idx.get(word, self.vocab.word2idx['<unk>']) for word in tokens]
        
        # Pad sequence
        padded_tokens = token_indices[:self.max_seq_length]
        padded_tokens += [self.vocab.word2idx['<pad>']] * (self.max_seq_length - len(padded_tokens))
        
        return padded_tokens

class Vocabulary:
    """A simple vocabulary class to map words to indices and vice-versa."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

if __name__ == '__main__':
    # Example of how to use the Dataset
    # This assumes you have run the data_simulator.py script first
    DATA_DIR = "../data"
    CSV_FILE = os.path.join(DATA_DIR, "metadata.csv")

    if not os.path.exists(CSV_FILE):
        print("Error: metadata.csv not found!")
        print("Please run src/data_simulator.py first to generate the dataset.")
    else:
        dataset = HealthcareMultimodalDataset(csv_file=CSV_FILE, root_dir=DATA_DIR)
        
        print(f"Dataset loaded successfully!")
        print(f"Number of samples: {len(dataset)}")
        print(f"Vocabulary size: {len(dataset.vocab)}")
        
        # Get a sample
        image, text, label = dataset[0]
        print("\n--- Sample Item ---")
        print(f"Image shape: {image.shape}")
        print(f"Text shape: {text.shape}")
        print(f"Label: {label.item()}")
        print(f"Sample text tokens: {text[:15]}...")
        print("---------------------")
