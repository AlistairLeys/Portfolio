"""
Training script for sentiment analysis models

This script trains sentiment analysis models on synthetic data and evaluates their performance.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from model import create_model
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def generate_synthetic_data(num_samples=10000):
    """
    Generate synthetic sentiment data for demonstration
    
    Args:
        num_samples: Number of samples to generate
    
    Returns:
        texts: List of text samples
        labels: Sentiment labels (0=negative, 1=positive, 2=neutral)
    """
    # Positive sentiment words and phrases
    positive_words = [
        'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'love',
        'perfect', 'best', 'awesome', 'outstanding', 'brilliant', 'superb',
        'delighted', 'happy', 'satisfied', 'recommend', 'impressed', 'pleased'
    ]
    
    # Negative sentiment words and phrases
    negative_words = [
        'terrible', 'awful', 'horrible', 'worst', 'bad', 'hate',
        'disappointing', 'poor', 'useless', 'waste', 'broken', 'defective',
        'unsatisfied', 'unhappy', 'frustrated', 'angry', 'disappointed', 'regret'
    ]
    
    # Neutral words and phrases
    neutral_words = [
        'okay', 'average', 'normal', 'standard', 'typical', 'regular',
        'acceptable', 'fine', 'moderate', 'adequate', 'fair', 'decent'
    ]
    
    # Product/service contexts
    contexts = [
        'product', 'service', 'item', 'purchase', 'experience', 'quality',
        'delivery', 'customer service', 'feature', 'performance', 'value'
    ]
    
    texts = []
    labels = []
    
    # Generate samples for each sentiment
    samples_per_class = num_samples // 3
    
    # Generate positive samples
    for _ in range(samples_per_class):
        sentiment_word = np.random.choice(positive_words)
        context = np.random.choice(contexts)
        templates = [
            f"This {context} is {sentiment_word}!",
            f"Really {sentiment_word} {context}.",
            f"I am very {sentiment_word} with this {context}.",
            f"The {context} exceeded my expectations. {sentiment_word}!",
            f"{sentiment_word.capitalize()} {context}, highly satisfied."
        ]
        text = np.random.choice(templates)
        texts.append(text)
        labels.append(1)  # Positive
    
    # Generate negative samples
    for _ in range(samples_per_class):
        sentiment_word = np.random.choice(negative_words)
        context = np.random.choice(contexts)
        templates = [
            f"This {context} is {sentiment_word}.",
            f"Really {sentiment_word} {context}.",
            f"I am very {sentiment_word} with this {context}.",
            f"The {context} was {sentiment_word}. Not recommended.",
            f"{sentiment_word.capitalize()} {context}, very disappointed."
        ]
        text = np.random.choice(templates)
        texts.append(text)
        labels.append(0)  # Negative
    
    # Generate neutral samples
    for _ in range(samples_per_class):
        sentiment_word = np.random.choice(neutral_words)
        context = np.random.choice(contexts)
        templates = [
            f"This {context} is {sentiment_word}.",
            f"The {context} is {sentiment_word}, nothing special.",
            f"It's an {sentiment_word} {context}.",
            f"The {context} meets basic expectations. {sentiment_word}.",
            f"{sentiment_word.capitalize()} {context}, as expected."
        ]
        text = np.random.choice(templates)
        texts.append(text)
        labels.append(2)  # Neutral
    
    return texts, labels


def plot_training_history(history, save_path='data/training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path='data/confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive', 'Neutral'],
                yticklabels=['Negative', 'Positive', 'Neutral'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main(args):
    """Main training function"""
    print("="*70)
    print("SENTIMENT ANALYSIS MODEL TRAINING")
    print("="*70)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate or load data
    print("\n1. Generating synthetic data...")
    texts, labels = generate_synthetic_data(args.num_samples)
    print(f"Generated {len(texts)} samples")
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    print(f"\n3. Creating {args.model} model...")
    model = create_model(
        model_type=args.model,
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        max_length=args.max_length,
        num_classes=3
    )
    
    print("\nModel Architecture:")
    model.get_summary()
    
    # Prepare sequences
    print("\n4. Preparing sequences...")
    model.create_tokenizer(X_train)
    X_train_seq = model.texts_to_sequences(X_train)
    X_val_seq = model.texts_to_sequences(X_val)
    X_test_seq = model.texts_to_sequences(X_test)
    
    # Convert labels to numpy arrays
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    # Train model
    print(f"\n5. Training model for {args.epochs} epochs...")
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\n6. Evaluating model...")
    test_loss, test_accuracy = model.model.evaluate(X_test_seq, y_test)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Predictions
    y_pred_probs = model.model.predict(X_test_seq)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Negative', 'Positive', 'Neutral']))
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Save model
    model_path = f'data/{args.model}_sentiment_model.h5'
    model.model.save(model_path)
    print(f"\n7. Model saved to {model_path}")
    
    # Example predictions
    print("\n8. Example predictions:")
    test_texts = [
        "This product is absolutely amazing!",
        "Terrible experience, very disappointed.",
        "It's okay, nothing special.",
        "Best purchase ever!",
        "Waste of money."
    ]
    
    test_sequences = model.texts_to_sequences(test_texts)
    predictions = model.model.predict(test_sequences)
    predicted_labels = np.argmax(predictions, axis=1)
    
    sentiment_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    
    for text, label, probs in zip(test_texts, predicted_labels, predictions):
        print(f"\nText: '{text}'")
        print(f"Predicted: {sentiment_map[label]}")
        print(f"Probabilities: Neg={probs[0]:.3f}, Pos={probs[1]:.3f}, Neu={probs[2]:.3f}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--model', type=str, default='lstm', 
                       choices=['lstm', 'gru', 'cnn_lstm', 'dense'],
                       help='Model architecture to use')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--vocab-size', type=int, default=10000,
                       help='Vocabulary size')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--max-length', type=int, default=100,
                       help='Maximum sequence length')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    
    args = parser.parse_args()
    main(args)
