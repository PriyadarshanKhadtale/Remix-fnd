"""
Text Classification Model
=========================
Neural network architecture for text-based fake news detection.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class TextClassifier(nn.Module):
    """
    Text classifier using pre-trained transformer encoder.
    
    Architecture:
        - Pre-trained RoBERTa/DistilRoBERTa encoder
        - Classification head (768 -> 256 -> 2)
    """
    
    def __init__(
        self, 
        model_name: str = "distilroberta-base",
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for RoBERTa
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Classify
        logits = self.classifier(cls_output)
        
        return logits
    
    def get_attention_weights(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Get attention weights for explainability."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        # Return last layer attention weights
        return outputs.attentions[-1]

