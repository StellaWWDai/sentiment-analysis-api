# Sentiment Analysis Model & REST API

This project contains a sentiment analysis machine learning classifier trained on 1.6 million tweets, along with a Django-based RESTful API that serves the model for inference.

## Project Structure

* `sentiment_analysis_*.ipynb`: Jupyter notebooks exploring different machine learning approaches (Logistic Regression, SVM, Naive Bayes, Random Forest, XGBoost).
* `sentiment_model/`: Reusable Python module containing the core `SentimentClassifier` class (`classifier.py`) and a training script (`train_model.py`).
* `models/`: Directory containing the saved artifacts (e.g., `model.joblib` and `vectorizer.joblib`) for the best model.
* `api/`: Django REST framework project to serve the `POST /evaluate` endpoint.

## 1. Environment Setup

It is highly recommended to use a virtual environment. The project manages dependencies via `requirements.txt`.

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (including Django and Django REST Framework)
pip install -r requirements.txt
```

## 1.1 Dataset Download

The dataset is not included in this repository due to size constraints. Download it from:

**Sentiment140 Dataset**: [Download from Google Drive](https://docs.google.com/file/d/0B04GJPshIjmPRnZManQwWEdTZjg/edit)

After downloading, extract and place the files in a `trainingandtestdata/` directory:

```
trainingandtestdata/
├── training.1600000.processed.noemoticon.csv   (Training data - 1.6M tweets)
└── testdata.manual.2009.06.14.csv              (Test data - 498 tweets)
```

**Note**: The pre-trained model files (`models/model.joblib` and `models/vectorizer.joblib`) are included in this repository, so you can run the API immediately without training.

## 2. Model Training (If needed)

The trained Logistic Regression model is chosen for deployment because it offers excellent baseline accuracy, outputs probabilities natively, and is computationally lightweight for a synchronous API.

The model files (`model.joblib` and `vectorizer.joblib`) are expected to be in the `models/` directory. If you need to retrain or update the model:

```bash
# Train on the full dataset and save to the models/ directory
python sentiment_model/train_model.py
```

*Note: You can also limit the number of training samples for faster iteration using the `--sample` flag (e.g., `python sentiment_model/train_model.py --sample 100000`).*

## 3. Running the REST API

The API handles requests and runs standard pre-processing before delegating to the model for predictions. The model loads into memory exactly once at application startup.

To start the server locally:

```bash
cd api
python manage.py runserver
```

The server runs on `http://127.0.0.1:8000/`.

## 4. Using the API

The REST API exposes the following endpoint to test sentences.

**Endpoint**: `POST /evaluate`
**Content-Type**: `application/json`

### Example Request

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"sentence": "I love this product!"}' \
     http://127.0.0.1:8000/evaluate
```

### Example Response (200 OK)

```json
{
    "sentiment": "positive",
    "confidence": 0.92,
    "probability": 0.92
}
```

The API will output:
* `"positive"` if internal probability > 0.6
* `"negative"` if internal probability < 0.4
* `"neutral"` for intermediate probabilities

## 5. Model Methodology & Discussion

This repository tests five different classifiers to solve the sentiment evaluation problem:
1. **Logistic Regression (TF-IDF)** — Fast inference, scales easily to 1.6M rows, easily outputs confidence thresholds. **Chosen for the API.**
2. **Support Vector Machines (LinearSVC + TF-IDF)** — Nearly identical to logistic regression but requires probability calibration.
3. **Naive Bayes (CountVectorizer)** — Probabilistically sound but slightly lower accuracy.
4. **Random Forest (TF-IDF)** — Explored for non-linear interactions, but required a sub-sampled dataset due to high memory consumption on 1.6M rows.
5. **XGBoost (TF-IDF)** — Strong ensemble learner, but same hardware limitations as Random Forest.

Logistic Regression proved to be the strongest balance of API performance, ease of deployment, and standard baseline accuracy (binary classification ~80%) for this dataset.
