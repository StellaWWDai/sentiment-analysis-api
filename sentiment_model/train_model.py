#!/usr/bin/env python3
"""
Training script for the Sentiment Classifier.

Usage:
    python train_model.py                    # Train on full dataset
    python train_model.py --sample 100000    # Train on 100k samples (faster)
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_model.classifier import SentimentClassifier


def main():
    parser = argparse.ArgumentParser(description='Train the sentiment classifier')
    parser.add_argument(
        '--data',
        type=str,
        default='trainingandtestdata/training.1600000.processed.noemoticon.csv',
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Number of samples to use (default: use all data)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='Directory to save the trained model'
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, args.data)
    model_dir = os.path.join(project_root, args.output)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    print("=" * 60)
    print("Sentiment Classifier Training")
    print("=" * 60)
    print(f"Data: {data_path}")
    print(f"Sample size: {args.sample or 'All data'}")
    print(f"Output: {model_dir}")
    print("=" * 60)

    # Train the model
    classifier = SentimentClassifier()
    metrics = classifier.train(data_path, sample_size=args.sample)

    # Save the model
    classifier.save(model_dir)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Training samples: {metrics['train_samples']:,}")
    print(f"Vocabulary size: {metrics['vocab_size']:,}")
    print(f"Model saved to: {model_dir}")

    # Test with some examples
    print("\n" + "=" * 60)
    print("Testing with sample sentences:")
    print("=" * 60)

    test_sentences = [
        "I love this product! It's amazing!",
        "This is terrible, worst experience ever.",
        "The weather is nice today.",
        "I hate waiting in long lines.",
        "Just had the best coffee of my life!"
    ]

    for sentence in test_sentences:
        result = classifier.evaluate(sentence)
        print(f"\n'{sentence}'")
        print(f"  -> {result['sentiment'].upper()} (confidence: {result['confidence']:.2%})")


if __name__ == '__main__':
    main()
