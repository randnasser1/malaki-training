"""
Text cleaning functions and dictionaries
"""

import re
import html
from typing import Dict, Optional, List, Union
from collections import Counter
import warnings

import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM



SLANG_DICT = {
    
    'asl': 'age sex location',
    'a/s/l': 'age sex location',
    'm/f': 'male or female',
    'f/m': 'female or male',
    's/l': 'sex location',
    
    
    'pic': 'picture',
    'pics': 'pictures',
    'photo': 'photo',
    'photos': 'photos',
    'snap': 'snapchat',
    'sc': 'snapchat',
    'kik': 'kik messenger',
    'vid': 'video',
    'vids': 'videos',
    
    
    'wyd': 'what you doing',
    'wya': 'where you at',
    'hmu': 'hit me up',
    'irl': 'in real life',
    'j4f': 'just for fun',
    
    
    'age': 'age',
    'sex': 'sex',
    'location': 'location',
    'single': 'single',
    
    
    'u': 'you',
    'ur': 'your',
    'r': 'are',
    'yr': 'your',
    'yrs': 'yours',
    'plz': 'please',
    'pls': 'please',
    'thx': 'thanks',
    'ty': 'thank you',
    'tyvm': 'thank you very much',
    'np': 'no problem',
    'yw': 'you are welcome',
    'idk': 'i do not know',
    'ik': 'i know',
    'dunno': 'do not know',
    'idc': 'i do not care',
    'tbh': 'to be honest',
    'imo': 'in my opinion',
    'imho': 'in my humble opinion',
    'fwiw': 'for what it is worth',
    'afaik': 'as far as i know',
    
    
    'lol': 'laughing out loud',
    'lmao': 'laughing my ass off',
    'lmfao': 'laughing my fucking ass off',
    'rofl': 'rolling on the floor laughing',
    'roflmao': 'rolling on the floor laughing my ass off',
    'lulz': 'laughs',
    'lel': 'laugh',
    'omg': 'oh my god',
    'omfg': 'oh my fucking god',
    'wtf': 'what the fuck',
    'wth': 'what the hell',
    'nvm': 'never mind',
    'jk': 'just kidding',
    'j/k': 'just kidding',
    'smh': 'shaking my head',
    
    
    'afk': 'away from keyboard',
    'brb': 'be right back',
    'gtg': 'got to go',
    'ttyl': 'talk to you later',
    'cya': 'see you',
    'fyi': 'for your information',
    
    
    'gonna': 'going to',
    'wanna': 'want to',
    'gotta': 'got to',
    'kinda': 'kind of',
    'sorta': 'sort of',
    'lemme': 'let me',
    'gimme': 'give me',
    'outta': 'out of',
    'tryna': 'trying to',
}

# Dictionary for single-letter expansions - but only for standalone letters
# Note: 'm' is NOT included here because it's usually part of "I'm"
SINGLE_LETTER_DICT = {
    'f': 'female',
    'u': 'you',
    'r': 'are',
    'c': 'see',
    'y': 'why',
    'b': 'be',
    'n': 'and',
    'k': 'okay',
}

# Common spelling corrections that are safe to do statically
# These are for very common, unambiguous cases
STATIC_SPELLING_DICT = {
    'teh': 'the',
    'hte': 'the',
    'adn': 'and',
    'nad': 'and',
    'jsut': 'just',
    'siad': 'said',
    'fomr': 'from',
    'waht': 'what',
    'yuo': 'you',
    'thier': 'their',
    'recieve': 'receive',
    'seperate': 'separate',
    'definately': 'definitely',
    'accomodate': 'accommodate',
    'goverment': 'government',
}

URL_RE = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
WWW_RE = re.compile(r'www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?')
TAG_RE = re.compile(r'<[^>]+>')
MD5_RE = re.compile(r'\b[a-f0-9]{32}\b', re.I)
HEX_RE = re.compile(r'\b[0-9a-f]{8,40}\b', re.I)
MENTION_RE = re.compile(r'@\w+')
AUTHOR_RE = re.compile(r'^\w+:\s*', re.M)
TIME_RE = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b', re.I)
DATE_RE = re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b')
REPEAT_RE = re.compile(r'(.)\1{2,}')
PUNCT_RE = re.compile(r'([!?.]){2,}')
PUNCT_SPACE_RE = re.compile(r'([,;:])\1+')
SPECIAL_RE = re.compile(r'[^\w\s.,;\'"()\-]')
SPACE_RE = re.compile(r'\s+')
WORD_RE = re.compile(r'\b\w+\b')
SINGLE_LETTER_RE = re.compile(r'\b([a-z])\b(?!\')')

KEYBOARD_PATTERNS = {
    re.compile(r'iapos;', re.I): "I'm",
    re.compile(r'youse', re.I): 'use',
    re.compile(r'pare', re.I): 'are',
    re.compile(r'woarek', re.I): 'work',
    re.compile(r'aree', re.I): 'are',
    re.compile(r'byout', re.I): 'but',
    re.compile(r'youre', re.I): 'your',
    re.compile(r'soyounds', re.I): 'sounds',
    re.compile(r'coareareect', re.I): 'correct',
    re.compile(r'\bhte\b', re.I): 'the',
    re.compile(r'\bwich\b', re.I): 'which',
    re.compile(r'\btehre\b', re.I): 'there',
    re.compile(r'\btahts\b', re.I): "that's",
    re.compile(r'hglsdfhglhreuh', re.I): '',
    re.compile(r'sdhfgilhsdflkghlfdghlfsd', re.I): '',
    re.compile(r'sdijfioashfusdghsdfughosdhfsdhg', re.I): '',
    re.compile(r'jaja+', re.I): 'haha',
    re.compile(r'cvbvcv', re.I): '',
    re.compile(r'\bwhcih\b', re.I): 'which',
    re.compile(r'\bna d\b', re.I): 'and',
}

CONTRACTIONS = {
    re.compile(r"\b(i'm|im)\b", re.I): "I am",
    re.compile(r"\b(i'll|ill)\b", re.I): "I will",
    re.compile(r"\b(i'd|id)\b", re.I): "I would",
    re.compile(r"\b(i've|ive)\b", re.I): "I have",
    re.compile(r"\b(don't|dont)\b", re.I): "do not",
    re.compile(r"\b(doesn't|doesnt)\b", re.I): "does not",
    re.compile(r"\b(won't|wont)\b", re.I): "will not",
    re.compile(r"\b(can't|cant)\b", re.I): "cannot",
    re.compile(r"\b(couldn't|couldnt)\b", re.I): "could not",
    re.compile(r"\b(wouldn't|wouldnt)\b", re.I): "would not",
    re.compile(r"\b(shouldn't|shouldnt)\b", re.I): "should not",
    re.compile(r"\b(wasn't|wasnt)\b", re.I): "was not",
    re.compile(r"\b(weren't|werent)\b", re.I): "were not",
    re.compile(r"\b(haven't|havent)\b", re.I): "have not",
    re.compile(r"\b(hasn't|hasnt)\b", re.I): "has not",
    re.compile(r"\b(hadn't|hadnt)\b", re.I): "had not",
    re.compile(r"\b(didn't|didnt)\b", re.I): "did not",
    re.compile(r"\b(isn't|isnt)\b", re.I): "is not",
    re.compile(r"\b(aren't|arent)\b", re.I): "are not",
    re.compile(r"\b(ain't|aint)\b", re.I): "is not",
    re.compile(r"\b(you're|youre)\b", re.I): "you are",
    re.compile(r"\b(you'll|youll)\b", re.I): "you will",
    re.compile(r"\b(you'd|youd)\b", re.I): "you would",
    re.compile(r"\b(you've|youve)\b", re.I): "you have",
    re.compile(r"\b(he's|hes)\b", re.I): "he is",
    re.compile(r"\b(he'll|hell)\b", re.I): "he will",
    re.compile(r"\b(he'd|hed)\b", re.I): "he would",
    re.compile(r"\b(she's|shes)\b", re.I): "she is",
    re.compile(r"\b(she'll|shell)\b", re.I): "she will",
    re.compile(r"\b(she'd|shed)\b", re.I): "she would",
    re.compile(r"\b(it's|its)\b", re.I): "it is",
    re.compile(r"\b(it'll|itll)\b", re.I): "it will",
    re.compile(r"\b(it'd|itd)\b", re.I): "it would",
    re.compile(r"\b(we're|were)\b", re.I): "we are",
    re.compile(r"\b(we'll|well)\b", re.I): "we will",
    re.compile(r"\b(we'd|wed)\b", re.I): "we would",
    re.compile(r"\b(we've|weve)\b", re.I): "we have",
    re.compile(r"\b(they're|theyre)\b", re.I): "they are",
    re.compile(r"\b(they'll|theyll)\b", re.I): "they will",
    re.compile(r"\b(they'd|theyd)\b", re.I): "they would",
    re.compile(r"\b(they've|theyve)\b", re.I): "they have",
}


class SpellingCorrector:
    """
    Hugging Face-based spelling correction.
    Uses a sequence-to-sequence model for context-aware spelling correction.
    """
    
    def __init__(self, model_name: str = "oliverguhr/spelling-correction-english-base", 
                 device: Optional[str] = None,
                 batch_size: int = 32):
        """
        Initialize the spelling corrector.
        
        Args:
            model_name: Hugging Face model name
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for processing
        """
        self.batch_size = batch_size
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading spelling correction model on {self.device}...")
        
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                batch_size=batch_size
            )
            
            print("✓ Spelling correction model loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Failed to load spelling correction model: {e}")
            print("Spelling correction disabled")
            self.model = None
            self.pipeline = None
    
    def correct(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Correct spelling in text(s).
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Corrected text(s)
        """
        if self.pipeline is None:
            # Return original texts if no model available
            return texts
        
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Skip empty texts
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and len(text.strip()) > 0:
                non_empty_indices.append(i)
                non_empty_texts.append(text)
        
        if not non_empty_texts:
            return texts[0] if single_input else texts
        
        try:
            # Run correction
            results = self.pipeline(
                non_empty_texts,
                max_length=512,
                do_sample=False,
                num_beams=1
            )
            
            corrected = [r['generated_text'] for r in results]
            
        except Exception as e:
            print(f"Warning: Spelling correction failed for batch: {e}")
            corrected = non_empty_texts
        
        # Reconstruct full list with original order
        result = []
        corrected_idx = 0
        for i in range(len(texts)):
            if i in non_empty_indices:
                result.append(corrected[corrected_idx])
                corrected_idx += 1
            else:
                result.append(texts[i])
        
        return result[0] if single_input else result


# Global spelling corrector instance (lazy-loaded)
_SPELLING_CORRECTOR = None

def get_spelling_corrector(force_reload: bool = False) -> Optional[SpellingCorrector]:
    """
    Get or create the global spelling corrector instance.
    
    Args:
        force_reload: Force reload the model
        
    Returns:
        SpellingCorrector instance or None if not available
    """
    global _SPELLING_CORRECTOR
    if _SPELLING_CORRECTOR is None or force_reload:
        _SPELLING_CORRECTOR = SpellingCorrector()
    return _SPELLING_CORRECTOR


def clean_text(text: str, 
               aggressive: bool = False, 
               slang_dict: Optional[Dict] = None,
               use_spelling_correction: bool = True,
               spelling_batch_size: int = 32) -> str:
    """
    Main text cleaning function with optional Hugging Face spelling correction.
    
    Args:
        text: Input text to clean
        aggressive: Whether to apply aggressive cleaning (contractions, etc.)
        slang_dict: Optional slang dictionary to use
        use_spelling_correction: Whether to use Hugging Face spelling correction
        spelling_batch_size: Batch size for spelling correction
    
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Unescape HTML
    text = html.unescape(text)
    
    # Lowercase (do this early for consistency)
    text = text.lower()
    
    # Remove URLs
    text = URL_RE.sub(' ', text)
    text = WWW_RE.sub(' ', text)
    
    # Remove HTML tags
    text = TAG_RE.sub(' ', text)
    
    # Remove hashes and hex
    text = MD5_RE.sub(' ', text)
    text = HEX_RE.sub(' ', text)
    
    # Remove author markers, mentions, time/date
    text = AUTHOR_RE.sub('', text)
    text = MENTION_RE.sub(' ', text)
    text = TIME_RE.sub(' ', text)
    text = DATE_RE.sub(' ', text)
    
    # Fix common keyboard patterns
    for pattern, replacement in KEYBOARD_PATTERNS.items():
        text = pattern.sub(replacement, text)
    
    # Apply static spelling corrections for very common cases
    for misspelled, correct in STATIC_SPELLING_DICT.items():
        pattern = re.compile(r'\b' + re.escape(misspelled) + r'\b')
        text = pattern.sub(correct, text)
    
    # Handle contractions FIRST (before single letters)
    if aggressive:
        for pattern, replacement in CONTRACTIONS.items():
            text = pattern.sub(replacement, text)
    
    # Apply slang dictionary (but avoid breaking contractions)
    if slang_dict:
        def replace_slang(match):
            word = match.group(0)
            # Don't replace words that look like they might be part of contractions
            if word in ["im", "id", "ill", "ive"] and aggressive:
                return word  # Let contractions handle these
            return slang_dict.get(word.lower(), word)
        text = WORD_RE.sub(replace_slang, text)
    
    # Handle single letters (f -> female, etc.)
    def replace_single_letter(match):
        letter = match.group(1)
        # Check if this might be part of a contraction
        context_before = text[max(0, match.start()-3):match.start()]
        context_after = text[match.end():match.end()+3]
        
        # Skip if it looks like it's part of a contraction
        if any(ctx in context_before + context_after for ctx in ["'", "im", "re", "ve", "ll", "d"]):
            return letter
            
        return SINGLE_LETTER_DICT.get(letter, letter)
    
    text = SINGLE_LETTER_RE.sub(replace_single_letter, text)
    
    # Apply Hugging Face spelling correction if requested
    if use_spelling_correction:
        corrector = get_spelling_corrector()
        if corrector is not None:
            text = corrector.correct(text)
    
    # Normalize repeated letters (but keep at least 2 for emphasis)
    def normalize_repeats(match):
        char = match.group(1)
        return char * 2  # Keep exactly 2 repetitions
    
    text = re.sub(r'(.)\1{3,}', normalize_repeats, text)
    
    # Remove excessive punctuation
    text = PUNCT_RE.sub(r'\1', text)
    text = PUNCT_SPACE_RE.sub(r'\1', text)
    
    # Clean up remaining special characters
    text = re.sub(r'[^a-z0-9\s.,!?;:\'\"()\-]', ' ', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])(?=[^\s])', r'\1 ', text)
    
    # Normalize whitespace
    text = SPACE_RE.sub(' ', text).strip()
    
    return text


def clean_text_batch(texts: List[str], 
                     aggressive: bool = False,
                     slang_dict: Optional[Dict] = None,
                     use_spelling_correction: bool = True,
                     batch_size: int = 32) -> List[str]:
    """
    Clean a batch of texts efficiently.
    
    Args:
        texts: List of texts to clean
        aggressive: Whether to apply aggressive cleaning
        slang_dict: Optional slang dictionary
        use_spelling_correction: Whether to use spelling correction
        batch_size: Batch size for spelling correction
        
    Returns:
        List of cleaned texts
    """
    # First apply all non-spelling cleaning to each text
    partially_cleaned = []
    for text in texts:
        cleaned = clean_text(
            text, 
            aggressive=aggressive, 
            slang_dict=slang_dict,
            use_spelling_correction=False  # Skip spelling correction for now
        )
        partially_cleaned.append(cleaned)
    
    # Then apply batch spelling correction if requested
    if use_spelling_correction:
        corrector = get_spelling_corrector()
        if corrector is not None:
            return corrector.correct(partially_cleaned)
    
    return partially_cleaned


def clean_text_series(series: pd.Series, 
                      aggressive: bool = False, 
                      slang_dict: Optional[Dict] = None,
                      use_spelling_correction: bool = True,
                      batch_size: int = 32) -> pd.Series:
    """
    Vectorized version of clean_text for pandas Series.
    
    Args:
        series: Pandas Series of texts
        aggressive: Whether to apply aggressive cleaning
        slang_dict: Optional slang dictionary
        use_spelling_correction: Whether to use spelling correction
        batch_size: Batch size for spelling correction
        
    Returns:
        Cleaned Series
    """
    texts = series.tolist()
    cleaned = clean_text_batch(
        texts, 
        aggressive=aggressive, 
        slang_dict=slang_dict,
        use_spelling_correction=use_spelling_correction,
        batch_size=batch_size
    )
    return pd.Series(cleaned, index=series.index)


def find_unusual_patterns(texts, known_dict=None, top_k=15):
    """Find patterns not covered by cleaning pipeline."""
    known_dict = known_dict or {**SLANG_DICT, **SINGLE_LETTER_DICT, **STATIC_SPELLING_DICT}
    known_words = set(known_dict.keys()) | set(known_dict.values())
    
    # Auto-fixed patterns we're aware of
    auto_fixed = {'im', 'dont', 'cant', 'teh', 'hte', 'adn', 'nad'}
    
    words = []
    for text in texts:
        if text and isinstance(text, str):
            words.extend(re.findall(r'\b[a-z]+\b', text.lower()))
    
    unknown = [w for w in words if w not in known_words and w not in auto_fixed and len(w) > 2]
    
    patterns = {
        'excessive_repeats': [w for w in unknown if re.search(r'(.)\1{3,}', w)],
        'moderate_repeats': [w for w in unknown if re.search(r'(.)\1{2}', w) 
                            and not re.search(r'(.)\1{3,}', w)],
        'keyboard_mash': [w for w in unknown if re.search(r'(qwe|wer|asd|sdf|zxc)', w)],
        'no_vowels': [w for w in unknown if len(w) > 4 and not re.search(r'[aeiouy]', w)],
    }
    
    print("\n=== UNUSUAL PATTERNS FOUND ===")
    for name, pattern_words in patterns.items():
        if pattern_words:
            print(f"\n{name.upper()}:")
            for word, count in Counter(pattern_words).most_common(top_k):
                print(f"  {word}: {count}")
    
    return patterns