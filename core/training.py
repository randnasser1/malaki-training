"""
Model training utilities
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os
from typing import Optional, Tuple, List, Dict
import numpy as np


class ChatDataset(Dataset):
    """PyTorch Dataset for text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def prepare_for_training(df, text_col='text', label_col='label', 
                         max_length=512, test_size=0.2, random_state=42,
                         batch_size=16):
    """
    Prepare data for transformer model training.
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        df[text_col].values, df[label_col].values,
        test_size=test_size, random_state=random_state, 
        stratify=df[label_col].values if len(np.unique(df[label_col])) > 1 else None
    )
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Create datasets
    train_dataset = ChatDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = ChatDataset(X_val, y_val, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, tokenizer


def train_epoch(model, dataloader, optimizer, device):
    """Single training epoch."""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return accuracy, f1


def train_model(train_loader, val_loader, num_labels=2, 
                epochs=3, lr=2e-5, device='cuda'):
    """Full training loop."""
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=num_labels
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")
    
    return model


def save_model(model, tokenizer, path: str):
    """Save model and tokenizer."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model saved to {path}")


def load_model(path: str, num_labels: int = 2):
    """Load model and tokenizer."""
    model = RobertaForSequenceClassification.from_pretrained(path)
    tokenizer = RobertaTokenizer.from_pretrained(path)
    return model, tokenizer