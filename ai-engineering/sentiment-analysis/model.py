"""
Sentiment Analysis Model Architectures

This module contains various neural network architectures for sentiment analysis.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


class SentimentModel:
    """Base class for sentiment analysis models"""
    
    def __init__(self, vocab_size=10000, embedding_dim=128, max_length=100, num_classes=3):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_classes = num_classes
        self.model = None
        self.tokenizer = None
        
    def create_tokenizer(self, texts):
        """Create and fit tokenizer on texts"""
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        return self.tokenizer
    
    def texts_to_sequences(self, texts):
        """Convert texts to padded sequences"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        return padded
    
    def build_model(self):
        """Build the model architecture (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_summary(self):
        """Print model summary"""
        return self.model.summary()


class LSTMSentimentModel(SentimentModel):
    """LSTM-based sentiment analysis model"""
    
    def build_model(self):
        """Build LSTM model"""
        self.model = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.SpatialDropout1D(0.2),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return self.model


class GRUSentimentModel(SentimentModel):
    """GRU-based sentiment analysis model"""
    
    def build_model(self):
        """Build GRU model"""
        self.model = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.SpatialDropout1D(0.2),
            layers.Bidirectional(layers.GRU(64, return_sequences=True)),
            layers.Bidirectional(layers.GRU(32)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return self.model


class CNNLSTMSentimentModel(SentimentModel):
    """CNN-LSTM hybrid sentiment analysis model"""
    
    def build_model(self):
        """Build CNN-LSTM hybrid model"""
        self.model = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.Conv1D(128, 5, activation='relu'),
            layers.MaxPooling1D(5),
            layers.Conv1D(64, 5, activation='relu'),
            layers.MaxPooling1D(5),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return self.model


class SimpleDenseModel(SentimentModel):
    """Simple dense network for baseline comparison"""
    
    def build_model(self):
        """Build simple dense model"""
        self.model = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return self.model


def create_model(model_type='lstm', vocab_size=10000, embedding_dim=128, max_length=100, num_classes=3):
    """
    Factory function to create sentiment analysis models
    
    Args:
        model_type: Type of model ('lstm', 'gru', 'cnn_lstm', 'dense')
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embedding layer
        max_length: Maximum sequence length
        num_classes: Number of sentiment classes
    
    Returns:
        SentimentModel instance
    """
    models_dict = {
        'lstm': LSTMSentimentModel,
        'gru': GRUSentimentModel,
        'cnn_lstm': CNNLSTMSentimentModel,
        'dense': SimpleDenseModel
    }
    
    if model_type not in models_dict:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models_dict.keys())}")
    
    model_class = models_dict[model_type]
    model = model_class(vocab_size, embedding_dim, max_length, num_classes)
    model.build_model()
    model.compile_model()
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Creating LSTM model...")
    lstm_model = create_model('lstm')
    print(lstm_model.get_summary())
    
    print("\nCreating GRU model...")
    gru_model = create_model('gru')
    print(gru_model.get_summary())
    
    print("\nCreating CNN-LSTM model...")
    cnn_lstm_model = create_model('cnn_lstm')
    print(cnn_lstm_model.get_summary())
