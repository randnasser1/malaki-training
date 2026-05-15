# Malaki Training

Model training notebooks and scripts for Malaki AI Child Guardian.

## Models Trained

| Model | Framework | Task | Accuracy |
|-------|-----------|------|----------|
| RoBERTa | Hugging Face Transformers | Grooming detection | 97.45% |
| BERT | Hugging Face Transformers | Author profiling (adult/minor) | 91.79% |
| Random Forest | Scikit-learn | Music mood classification | 93.07% |
| DistilBERT | Hugging Face Transformers | Emotion detection | - |
| TBATS | Statsmodels | Behavioral anomaly detection | - |

## Setup

1. Clone repository

2. Install dependencies
pip install -r requirements.txt

3. Download datasets

## Notebooks

| File | Description |
|------|-------------|
| predator_training.ipynb | Fine-tune RoBERTa for grooming detection |
| author_profiling_bert.ipynb | Fine-tune BERT for age classification |
| random_forest_music.ipynb | Train RF on Spotify audio features |
| distilbert_emotions_training.ipynb | Fine-tune DistilBERT for emotion detection |
| tbats_app.ipynb | Time series app usage anomaly detection |
| tbats_music.ipynb | Time series music activity anomaly detection |


## Output Models

Export trained models to backend/models/ directory:
- predator_roberta/ (saved model folder)
- author_bert/ (saved model folder)
- random_forest_emotion_model.pkl
- emotions_distilbert/ (saved model folder)

## Datasets

Grooming Detection: PAN 2012 
Author Profiling: PAN 13 + Pan 14
Music Mood: Spotify API + ReccoBeats API dataset of audio features
Emotion Detection: GoEmotions 
Behavioral: Synthetic time series data (30-day patterns)