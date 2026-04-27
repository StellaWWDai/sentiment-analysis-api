"""
Sentiment Analysis Classifier

This module implements a sentiment classifier for Twitter data using TF-IDF
vectorization and Logistic Regression. This approach was chosen for:
- Simplicity and interpretability
- Fast training and inference
- Good baseline performance on short text like tweets
- Easy to deploy and maintain

Alternative approaches considered:
- BERT/DistilBERT: Higher accuracy but much slower, requires GPU for practical use
- LSTM/RNN: Good for sequences but slower training, more complex
- Naive Bayes: Simpler but typically lower accuracy than Logistic Regression
"""

import re
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class SentimentClassifier:
    """
    A sentiment classifier that uses TF-IDF + Logistic Regression.

    Attributes:
        vectorizer: TF-IDF vectorizer for text encoding
        model: Logistic Regression classifier
        is_trained: Boolean indicating if model has been trained
        positive_threshold: Probability above which sentiment is positive (default: 0.6)
        negative_threshold: Probability below which sentiment is negative (default: 0.4)

    Sentiment Classification Logic:
        - P(positive) > 0.6  → "positive"
        - P(positive) < 0.4  → "negative"
        - 0.4 <= P(positive) <= 0.6 → "neutral" (model is uncertain)
    """

    # Configurable thresholds for sentiment classification
    POSITIVE_THRESHOLD = 0.6
    NEGATIVE_THRESHOLD = 0.4

    def __init__(self):
        """Initialize the classifier with default parameters."""
        self.vectorizer = TfidfVectorizer(
            max_features=10000,      # Limit vocabulary size for efficiency
            ngram_range=(1, 2),      # Use unigrams and bigrams
            min_df=2,                # Ignore rare terms
            max_df=0.95              # Ignore very common terms
        )
        self.model = LogisticRegression(
            max_iter=1000,
            C=1.0,                   # Regularization strength
            random_state=42          # For reproducibility
        )
        self.is_trained = False

    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess tweet text.

        Args:
            text: Raw tweet text

        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove @mentions
        text = re.sub(r'@\w+', '', text)

        # Remove hashtag symbol but keep the text
        text = re.sub(r'#', '', text)

        # Remove special characters and digits (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def train(self, data_path: str, sample_size: int = None) -> dict:
        """
        Train the sentiment classifier on the Twitter dataset.

        Trains on 100% of the training data. Use the separate test file
        for evaluation (testdata.manual.2009.06.14.csv).

        Args:
            data_path: Path to the training CSV file
            sample_size: Optional limit on number of samples (for faster training)

        Returns:
            Dictionary containing training metrics
        """
        print("Loading data...")

        # Load CSV without headers (Twitter sentiment format)
        # Columns: sentiment, id, date, query, user, text
        df = pd.read_csv(
            data_path,
            encoding='latin-1',
            header=None,
            names=['sentiment', 'id', 'date', 'query', 'user', 'text']
        )

        # Sample data if specified (useful for faster iteration)
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Using {sample_size} samples for training")

        print(f"Total samples: {len(df)}")

        # Convert sentiment: 0 -> 0 (negative), 4 -> 1 (positive)
        df['label'] = df['sentiment'].apply(lambda x: 1 if x == 4 else 0)

        # Preprocess text
        print("Preprocessing text...")
        df['clean_text'] = df['text'].apply(self.preprocess_text)

        # Remove empty texts
        df = df[df['clean_text'].str.len() > 0]
        print(f"Training samples after preprocessing: {len(df)}")

        # Fit TF-IDF vectorizer and transform training data
        print("Vectorizing text with TF-IDF...")
        X_train = self.vectorizer.fit_transform(df['clean_text'])
        y_train = df['label'].values

        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        # Train the model on ALL data
        print("Training Logistic Regression model...")
        self.model.fit(X_train, y_train)

        print("\nTraining complete!")

        self.is_trained = True

        return {
            'train_samples': len(df),
            'vocab_size': len(self.vectorizer.vocabulary_)
        }

    def evaluate(self, sentence: str) -> dict:
        """
        Evaluate the sentiment of a given sentence.

        Uses probability thresholds to determine sentiment:
        - P(positive) > 0.6  → "positive"
        - P(positive) < 0.4  → "negative"
        - Otherwise → "neutral" (model is uncertain)

        Args:
            sentence: Input text to analyze

        Returns:
            Dictionary containing:
                - sentiment: 'positive', 'negative', or 'neutral'
                - confidence: Probability score (0-1) for the predicted class
                - probability: Raw probability for positive class
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first or load a saved model.")

        # Preprocess the input
        clean_text = self.preprocess_text(sentence)

        if not clean_text:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'probability': 0.5,
                'error': 'Input text is empty after preprocessing'
            }

        # Vectorize
        text_tfidf = self.vectorizer.transform([clean_text])

        # Get probability of positive class
        probabilities = self.model.predict_proba(text_tfidf)[0]
        positive_prob = probabilities[1]  # P(positive)

        # Apply threshold logic for sentiment classification
        if positive_prob > self.POSITIVE_THRESHOLD:
            sentiment = 'positive'
            confidence = positive_prob
        elif positive_prob < self.NEGATIVE_THRESHOLD:
            sentiment = 'negative'
            confidence = 1 - positive_prob  # P(negative)
        else:
            sentiment = 'neutral'
            # Confidence for neutral = how close to 0.5 (center of uncertainty)
            confidence = 1 - abs(positive_prob - 0.5) * 2

        return {
            'sentiment': sentiment,
            'confidence': round(float(confidence), 4),
            'probability': round(float(positive_prob), 4)
        }

    def save(self, model_dir: str) -> None:
        """
        Save the trained model and vectorizer to disk.

        Args:
            model_dir: Directory to save model artifacts
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model.")

        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(self.vectorizer, os.path.join(model_dir, 'vectorizer.joblib'))
        joblib.dump(self.model, os.path.join(model_dir, 'model.joblib'))

        print(f"Model saved to {model_dir}")

    def load(self, model_dir: str) -> None:
        """
        Load a trained model and vectorizer from disk.

        Args:
            model_dir: Directory containing model artifacts
        """
        vectorizer_path = os.path.join(model_dir, 'vectorizer.joblib')
        model_path = os.path.join(model_dir, 'model.joblib')

        if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model files not found in {model_dir}")

        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        self.is_trained = True

        print(f"Model loaded from {model_dir}")


# Convenience functions for standalone usage
def train(data_path: str, model_dir: str = None, sample_size: int = None) -> SentimentClassifier:
    """
    Train a new sentiment classifier.

    Args:
        data_path: Path to training CSV file
        model_dir: Optional directory to save the model
        sample_size: Optional limit on training samples

    Returns:
        Trained SentimentClassifier instance
    """
    classifier = SentimentClassifier()
    classifier.train(data_path, sample_size=sample_size)

    if model_dir:
        classifier.save(model_dir)

    return classifier


def evaluate(sentence: str, model_dir: str) -> dict:
    """
    Evaluate sentiment of a sentence using a saved model.

    Args:
        sentence: Text to analyze
        model_dir: Directory containing saved model

    Returns:
        Sentiment prediction dictionary
    """
    classifier = SentimentClassifier()
    classifier.load(model_dir)
    return classifier.evaluate(sentence)
