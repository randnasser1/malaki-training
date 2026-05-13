"""
Dataset-specific loaders for all Kaggle datasets
"""

import pandas as pd
import xml.etree.ElementTree as ET
import os
import json
from bs4 import BeautifulSoup
import re
from typing import Optional, Dict, List, Tuple

def load_pan12(xml_path: str, ids_path: str) -> pd.DataFrame:
    """Load PAN12 dataset."""
    with open(ids_path) as f:
        predator_ids = set(line.strip() for line in f.readlines())
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    rows = []
    for conv in root.findall(".//conversation"):
        conv_id = conv.attrib.get("id", "unknown")
        
        for msg in conv.findall(".//message"):
            text = msg.find("text").text or ""
            author = msg.find("author").text or "unknown"
            time = msg.find("time").text or ""
            
            label = 1 if author in predator_ids else 0
            
            rows.append({
                "conversation_id": conv_id,
                "author": author,
                "time": time,
                "text": text,
                "label": label,
                "source": "PAN12"
            })
    
    return pd.DataFrame(rows)

def load_pan14(xml_dir: str, dataset_type: str = "unknown", 
               truth_file: Optional[str] = None) -> pd.DataFrame:
    """Load PAN14 dataset - FIXED VERSION"""
    
    # Load truth mappings
    truth_mapping = {}
    if truth_file and os.path.exists(truth_file):
        print(f"Loading truth file: {truth_file}")
        with open(truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # PAN14 format: author_id:::gender:::age_group
                parts = line.split(":::")
                if len(parts) >= 3:
                    author_id = parts[0].strip()
                    gender = parts[1].strip()
                    age_group = parts[2].strip()
                    
                    # Clean values
                    gender = None if gender in ['', 'na', 'unknown'] else gender
                    age_group = None if age_group in ['', 'na', 'unknown'] else age_group
                    
                    truth_mapping[author_id] = {
                        'gender': gender,
                        'age_group': age_group
                    }
        
        print(f"✅ Loaded {len(truth_mapping)} author mappings from truth file")
    else:
        print(f"⚠️ Truth file not found: {truth_file}")

    rows = []
    files_processed = 0
    matched_count = 0
    
    for fname in os.listdir(xml_dir):
        if not fname.lower().endswith(".xml"):
            continue
            
        path = os.path.join(xml_dir, fname)
        files_processed += 1
        
        # Extract author_id from filename (PAN14 uses UUID as filename without extension)
        author_id = os.path.splitext(fname)[0]
        
        # Get gender/age from truth mapping
        gender = None
        age_group = None
        if author_id in truth_mapping:
            gender = truth_mapping[author_id].get('gender')
            age_group = truth_mapping[author_id].get('age_group')
            matched_count += 1
        
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            
            # PAN14 has different structure - look for all text content
            # Method 1: Find all document elements
            documents = root.findall(".//document")
            
            # Method 2: If no documents, try to get text from root
            if not documents:
                # Try to get text directly from XML
                text = ET.tostring(root, encoding='unicode', method='text')
                if text.strip():
                    rows.append({
                        "author_id": author_id,
                        "doc_id": f"doc_{files_processed}",
                        "text": text.strip(),
                        "gender": gender,
                        "age_group": age_group,
                        "dataset_type": dataset_type
                    })
            else:
                for doc in documents:
                    text = doc.text or ""
                    if text.strip():
                        # Clean HTML if present
                        if '<' in text and '>' in text:
                            text = BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
                        
                        doc_id = doc.attrib.get("id", f"doc_{len(rows)}")
                        
                        rows.append({
                            "author_id": author_id,
                            "doc_id": doc_id,
                            "text": text.strip(),
                            "gender": gender,
                            "age_group": age_group,
                            "dataset_type": dataset_type
                        })
        except Exception as e:
            print(f"⚠️ Error processing {fname}: {e}")
            continue
    
    print(f"Processed {files_processed} XML files")
    print(f"Matched {matched_count} authors to truth file ({matched_count/files_processed*100:.1f}%)")
    print(f"Created {len(rows)} text rows")
    
    return pd.DataFrame(rows)


def load_pan13(xml_dir: str, dataset_type: str = "unknown", 
               truth_file: Optional[str] = None, lang_filter: str = 'en') -> pd.DataFrame:
    """Load PAN13 dataset - FIXED VERSION"""
    
    def clean_attr(value):
        if value and value.strip().lower() not in {"xxx", "na", "unknown", ""}:
            return value.strip()
        return None
    
    # Load truth labels
    truth_mapping = {}
    if truth_file and os.path.exists(truth_file):
        print(f"Loading PAN13 truth file: {truth_file}")
        with open(truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # PAN13 format: author_id:::gender:::age_group (optional category)
                parts = line.split(':::')
                if len(parts) >= 3:
                    author_id = parts[0].strip()
                    gender = parts[1].strip()
                    age_group = parts[2].strip()
                    category = parts[3].strip() if len(parts) >= 4 else None
                    
                    truth_mapping[author_id] = {
                        'gender': gender if gender not in ['', 'na', 'unknown'] else None,
                        'age_group': age_group if age_group not in ['', 'na', 'unknown'] else None,
                        'category': category
                    }
        
        print(f"✅ Loaded {len(truth_mapping)} author mappings from truth file")
    
    rows = []
    files_processed = 0
    matched_count = 0
    
    for fname in os.listdir(xml_dir):
        if not fname.lower().endswith(".xml"):
            continue
        
        path = os.path.join(xml_dir, fname)
        files_processed += 1
        
        # Extract author_id (UUID before first underscore)
        author_id = fname.split('_')[0]
        
        # Try to extract metadata from filename
        parts = fname.replace('.xml', '').split('_')
        if len(parts) >= 4:
            filename_lang = parts[1]
            filename_age = parts[2] if parts[2] != 'XXX' else None
            filename_gender = parts[3].upper() if parts[3] != 'XXX' else None
        else:
            filename_lang = filename_age = filename_gender = None
        
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            
            # Get metadata from XML
            author_lang = clean_attr(root.attrib.get("lang")) or filename_lang
            author_gender = clean_attr(root.attrib.get("gender")) or filename_gender
            author_age = clean_attr(root.attrib.get("age_group")) or filename_age
            
            # Override with truth mapping if available
            category = None
            if author_id in truth_mapping:
                truth_data = truth_mapping[author_id]
                author_gender = truth_data['gender'] or author_gender
                author_age = truth_data['age_group'] or author_age
                category = truth_data['category']
                matched_count += 1
            
            # Language filter
            if lang_filter and author_lang and author_lang != lang_filter:
                continue
            
            # Extract conversations
            conversations = root.findall(".//conversation")
            for conv in conversations:
                text = conv.text or ""
                if text.strip():
                    # Clean text
                    text = BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
                    
                    # Create age_group display with category if present
                    age_group_display = author_age
                    if category:
                        age_group_display = f"{author_age}:::{category}" if author_age else category
                    
                    rows.append({
                        "author_id": author_id,
                        "doc_id": conv.attrib.get("id", f"conv_{len(rows)}"),
                        "text": text,
                        "gender": author_gender,
                        "age_group": age_group_display,
                        "age_group_base": author_age,
                        "category": category,
                        "dataset_type": dataset_type
                    })
        except Exception as e:
            print(f"⚠️ Error processing {fname}: {e}")
            continue
    
    print(f"Processed {files_processed} XML files")
    print(f"Matched {matched_count} authors to truth file ({matched_count/files_processed*100:.1f}%)")
    print(f"Created {len(rows)} conversation rows")
    
    return pd.DataFrame(rows)

def load_bf_psr(json_path: str) -> pd.DataFrame:
    """Load BF-PSR framework dataset."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rows = []
    for conv in data['conversation']:
        conv_id = conv['id']
        label = int(conv['label'])
        source = conv['source']
        
        for msg in conv['messages']:
            if msg.get('text'):
                rows.append({
                    "conversation_id": f"{source}_{conv_id}",
                    "source": source,
                    "label": label,
                    "author": msg.get('author', None),
                    "time": msg.get('time', None),
                    "text": msg['text']
                })
    
    df = pd.DataFrame(rows)
    return df.sort_values(by=['conversation_id', 'time']).reset_index(drop=True)


def load_goemotions(base_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load GoEmotions dataset."""
    df1 = pd.read_csv(f"{base_path}/goemotions_1.csv")
    df2 = pd.read_csv(f"{base_path}/goemotions_2.csv")
    df3 = pd.read_csv(f"{base_path}/goemotions_3.csv")
    
    df = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Clean and prepare
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ""].reset_index(drop=True)
    
    # Drop non-feature columns
    drop_cols = ['id', 'author', 'subreddit', 'link_id', 'parent_id', 
                 'created_utc', 'rater_id']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Get emotion columns (all 0/1 columns except 'text')
    emotion_cols = [col for col in df.columns if col != 'text' 
                   and df[col].dropna().isin([0, 1]).all()]
    
    # Filter rows with at least one emotion
    df = df[df[emotion_cols].sum(axis=1) > 0].reset_index(drop=True)
    
    return df, emotion_cols