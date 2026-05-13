"""
Utility functions for checkpointing and I/O
"""

import os
import pandas as pd
import pickle
import json
from typing import Optional, Union, Any

def save_checkpoint(data: Union[pd.DataFrame, Any], 
                    path: str, format: str = 'parquet'):
    """Save DataFrame checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if format == 'parquet':
        if isinstance(data, pd.DataFrame):
            data.to_parquet(path, index=False)
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
    elif format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False)
        else:
            with open(path, 'w') as f:
                json.dump(data, f)
    elif format == 'pickle':
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    print(f"Checkpoint saved to {path}")

def load_checkpoint(path: str, format: str = 'parquet'):
    """Load DataFrame checkpoint."""
    if not os.path.exists(path):
        return None
    
    if format == 'parquet':
        return pd.read_parquet(path)
    elif format == 'csv':
        return pd.read_csv(path)
    elif format == 'pickle':
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    return None

def save_json(data: Any, path: str):
    """Save data as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"JSON saved to {path}")


def load_json(path: str):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def print_dataset_stats(df, name="Dataset"):
    """Print basic statistics about a dataset."""
    print(f"\n{name} Statistics:")
    print(f"- Shape: {df.shape}")
    print(f"- Columns: {df.columns.tolist()}")
    
    if 'label' in df.columns:
        print(f"- Label distribution:")
        print(df['label'].value_counts())
    
    if 'text' in df.columns:
        print(f"- Text length stats:")
        print(df['text'].str.len().describe())
    
    return df