"""
Text Preprocessor
=================
Clean and tokenize text for the model.
"""

import re
from transformers import AutoTokenizer
from typing import Dict, Any


class TextPreprocessor:
    """Handles text cleaning and tokenization."""
    
    def __init__(self, model_name: str = "distilroberta-base", max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def clean_text(self, text: str) -> str:
        """
        Clean input text.
        
        - Remove extra whitespace
        - Remove URLs
        - Normalize unicode
        """
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Basic normalization
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> Dict[str, Any]:
        """
        Tokenize text for the model.
        
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        text = self.clean_text(text)
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
    
    def get_tokens(self, text: str):
        """Get individual tokens for a text (for explainability)."""
        text = self.clean_text(text)
        tokens = self.tokenizer.tokenize(text)
        return tokens

