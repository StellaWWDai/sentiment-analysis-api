# Interview Cramming Notes: Sentiment Analysis Project

## 1. PROJECT OVERVIEW

### What You Built
- **Sentiment classifier** trained on 1.6M tweets (Sentiment140 dataset)
- **REST API** using Django + Django REST Framework
- **Endpoint**: `POST /evaluate` - takes a sentence, returns sentiment

### Dataset Facts (Memorize These!)
| Fact | Value |
|------|-------|
| Training samples | 1,600,000 tweets |
| Test samples | 498 tweets |
| Training labels | 0 (negative), 4 (positive) - **NO NEUTRAL** |
| Test labels | 0, 2, 4 (includes neutral) |
| Avg tweet length | ~13 words |
| Class balance | 50/50 (800k each) |

---

## 2. MODEL CHOICE: WHY LOGISTIC REGRESSION?

### Your Answer
> "I chose **TF-IDF + Logistic Regression** because it provides an excellent balance of accuracy, interpretability, and deployment simplicity for this use case."

### Key Reasons (Memorize These!)

1. **Native Probability Output**
   - Logistic Regression outputs probabilities directly
   - Essential for our neutral threshold logic (0.4-0.6 = uncertain = neutral)
   - SVM requires additional calibration (CalibratedClassifierCV)

2. **Fast Training & Inference**
   - Trains on 1.6M samples in ~1-2 minutes
   - Inference: <1ms per prediction
   - Perfect for synchronous API (requirement says "handle synchronously")

3. **Interpretability**
   - Can inspect feature weights to understand predictions
   - Example: "sad" has weight -12.6, "cant wait" has weight +8.5
   - Easy to debug and explain to stakeholders

4. **Scalability**
   - Handles 1.6M samples without memory issues
   - Random Forest/XGBoost required subsampling due to memory

5. **Good Baseline Accuracy**
   - ~80% binary accuracy (positive vs negative)
   - Competitive with more complex models on this dataset

### Comparison Table (IMPORTANT - They Will Ask!)

| Model | Binary Acc | Pros | Cons |
|-------|-----------|------|------|
| **Logistic Regression** | ~80% | Fast, interpretable, native probs | Linear decision boundary |
| **SVM (LinearSVC)** | ~80% | Max-margin, good generalization | Needs calibration for probs |
| **Naive Bayes** | ~77% | Very fast, probabilistic | Independence assumption |
| **Random Forest** | ~78% | Non-linear, feature importance | Memory-heavy on 1.6M rows |
| **XGBoost** | ~79% | Powerful ensemble | Slow training, complex tuning |
| **BERT/Transformers** | ~85-90% | State-of-the-art | Requires GPU, slow inference |

### Why NOT Deep Learning/BERT?
> "For a synchronous REST API serving real-time requests, Logistic Regression provides sub-millisecond inference. BERT would add ~100-500ms latency per request and require GPU infrastructure. The ~5-10% accuracy gain doesn't justify the complexity for this use case."

---

## 3. FEATURE ENGINEERING: TF-IDF

### What is TF-IDF?
**Term Frequency - Inverse Document Frequency**

- **TF**: How often a word appears in THIS document
- **IDF**: How rare the word is across ALL documents
- **TF-IDF = TF × IDF**: Words that are frequent in one doc but rare overall get high scores

### The Formulas (Know These!)

**1. Term Frequency (TF)**
```
TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)
```
Or simply: `TF(t, d) = count(t in d) / len(d)`

**2. Inverse Document Frequency (IDF)**
```
IDF(t) = log(N / df(t))

Where:
- N = Total number of documents in corpus
- df(t) = Number of documents containing term t
```

**3. TF-IDF Score**
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

### Intuition Behind IDF

| Term | Documents containing it | IDF | Meaning |
|------|------------------------|-----|---------|
| "the" | 1,600,000 (all) | log(1.6M/1.6M) = 0 | Useless - appears everywhere |
| "happy" | 50,000 | log(1.6M/50K) = 3.5 | Somewhat distinctive |
| "ecstatic" | 500 | log(1.6M/500) = 8.1 | Very distinctive |

### Example Calculation

Document: "I love love love this product"
Corpus: 1,600,000 documents

| Term | TF | df(t) | IDF | TF-IDF |
|------|-----|-------|-----|--------|
| "love" | 3/6 = 0.5 | 100,000 | log(1.6M/100K) = 2.77 | 0.5 × 2.77 = **1.39** |
| "product" | 1/6 = 0.17 | 200,000 | log(1.6M/200K) = 2.08 | 0.17 × 2.08 = **0.35** |
| "i" | 1/6 = 0.17 | 1,500,000 | log(1.6M/1.5M) = 0.06 | 0.17 × 0.06 = **0.01** |

→ "love" gets the highest score because it's frequent in this doc AND distinctive across corpus

### Scikit-learn's TF-IDF Variant
Scikit-learn uses a slightly modified formula:
```
IDF(t) = log((N + 1) / (df(t) + 1)) + 1
```
The +1 terms prevent division by zero and ensure non-zero IDF values.

### Why TF-IDF Works for Sentiment
> "TF-IDF automatically downweights common words like 'the' and 'is' while boosting sentiment-specific words like 'love', 'hate', 'amazing'. This is why we don't need explicit stopword removal."

### Your TF-IDF Settings (Know These!)
```python
TfidfVectorizer(
    max_features=10000,    # Top 10k words/phrases
    ngram_range=(1, 2),    # Unigrams + Bigrams
    min_df=2,              # Ignore words appearing < 2 times
    max_df=0.95            # Ignore words in > 95% of docs
)
```

### Why These Settings?

| Parameter | Value | Why |
|-----------|-------|-----|
| `max_features=10000` | Limits vocabulary | Prevents overfitting, reduces memory |
| `ngram_range=(1,2)` | Bigrams | Captures "not good", "cant wait" |
| `min_df=2` | Min frequency | Removes typos, rare words |
| `max_df=0.95` | Max frequency | Removes "the", "a", "is" |

### Why Bigrams Matter (Great Interview Point!)
Without bigrams:
- "not" = negative feature
- "good" = positive feature
- "not good" → incorrectly predicted as mixed/neutral

With bigrams:
- "not good" is learned as a single negative feature
- "cant wait" is learned as positive (anticipation)
- "not bad" is learned as positive (understatement)

---

## 4. PREPROCESSING DECISIONS

### Your Preprocessing Pipeline
```python
1. Lowercase
2. Remove URLs (http://...)
3. Remove @mentions (@user)
4. Remove # symbol (keep hashtag text)
5. Remove special characters
6. Normalize whitespace
```

### Why NO Stopword Removal?
> "I intentionally kept stopwords because words like 'not', 'no', 'never' are critical for sentiment. Removing them would lose negation context."

**Evidence**: Tested with stopword removal → accuracy dropped from 77.5% to 75.7%

### Why NO Lemmatization?
> "Lemmatization adds complexity with minimal gain. TF-IDF with bigrams already captures most patterns. Also, sentiment is often in word form: 'loving' vs 'love' may have different intensities."

---

## 5. HANDLING NEUTRAL SENTIMENT

### The Problem
- Training data: Only positive (4) and negative (0)
- Test data: Has neutral (2) as well
- **You can't learn what you've never seen!**

### Your Solution: Probability Thresholds
```
P(positive) > 0.6  → "positive"   (model is confident positive)
P(positive) < 0.4  → "negative"   (model is confident negative)
0.4 ≤ P ≤ 0.6      → "neutral"    (model is uncertain)
```

### Why This Works
> "When the model outputs ~0.5 probability, it's essentially saying 'I can't tell if this is positive or negative.' That uncertainty maps naturally to neutral sentiment."

### Results by Class
| True Label | Accuracy | Explanation |
|------------|----------|-------------|
| Positive | 83% | High - trained on this |
| Negative | 63% | Good - trained on this |
| Neutral | 22% | Low - expected! Never trained on it |

### Interview Answer for Low Neutral Accuracy
> "The low neutral accuracy is expected and acceptable. We're using uncertainty as a proxy for neutral, not learning neutral features. To improve this, we'd need labeled neutral training data."

---

## 6. API FUNDAMENTALS (For Beginners)

### What is an API?
**API = Application Programming Interface**

An API is a way for two programs to talk to each other. In our case:
- **Client** (user's app, curl, browser) sends a request
- **Server** (our Django app) processes it and sends a response

Think of it like a restaurant:
- You (client) → give order to waiter (API) → kitchen (server) processes → waiter brings food back

### What is REST?
**REST = REpresentational State Transfer**

REST is a set of rules for designing APIs. Key principles:

| Principle | Meaning | Our Implementation |
|-----------|---------|-------------------|
| **Stateless** | Server doesn't remember previous requests | Each `/evaluate` call is independent |
| **Client-Server** | Client and server are separate | Django server + any client |
| **Uniform Interface** | Consistent URL patterns | `/evaluate` endpoint |
| **Resource-Based** | URLs represent resources | Sentiment evaluation is a resource |

### HTTP Methods (Verbs)
| Method | Purpose | Example |
|--------|---------|---------|
| **GET** | Retrieve data | `GET /users/123` (get user info) |
| **POST** | Create/Submit data | `POST /evaluate` (submit sentence) |
| **PUT** | Update entire resource | `PUT /users/123` (update user) |
| **PATCH** | Partial update | `PATCH /users/123` (update email only) |
| **DELETE** | Remove resource | `DELETE /users/123` |

**Why we use POST for `/evaluate`:**
> "We're submitting data (a sentence) to create a new prediction. POST is for creating resources or submitting data for processing."

### HTTP Status Codes
| Code | Meaning | When We Use It |
|------|---------|----------------|
| **200 OK** | Success | Sentiment returned successfully |
| **400 Bad Request** | Client error | Missing/invalid sentence |
| **404 Not Found** | Resource not found | Wrong URL |
| **405 Method Not Allowed** | Wrong HTTP method | Used GET instead of POST |
| **500 Internal Server Error** | Server crashed | Bug in our code |
| **503 Service Unavailable** | Server not ready | Model failed to load |

### Request/Response Cycle (Visual)
```
┌─────────────┐     HTTP POST /evaluate      ┌─────────────┐
│             │  ─────────────────────────▶  │             │
│   CLIENT    │  {"sentence": "I love it"}   │   SERVER    │
│  (curl/app) │                              │  (Django)   │
│             │  ◀─────────────────────────  │             │
└─────────────┘   {"sentiment": "positive"}  └─────────────┘
                        HTTP 200 OK
```

### JSON Format
**JSON = JavaScript Object Notation**

JSON is how we format data in API requests/responses:
```json
{
    "key": "value",
    "number": 123,
    "boolean": true,
    "array": ["a", "b", "c"],
    "nested": {"inner": "value"}
}
```

Our API uses JSON for both request and response.

---

## 6.1 OUR API ENDPOINT

### Endpoint Details
```
POST /evaluate
Content-Type: application/json

Request:  {"sentence": "I love this product!"}
Response: {"sentiment": "positive", "confidence": 0.92, "probability": 0.92}
```

### RESTful Principles Applied

| Principle | Implementation |
|-----------|----------------|
| **Proper HTTP Methods** | POST for creating a prediction |
| **Status Codes** | 200 OK, 400 Bad Request, 500 Error, 503 Service Unavailable |
| **JSON Format** | Request and response both JSON |
| **Stateless** | Each request is independent |
| **Resource-Oriented** | `/evaluate` is the evaluation resource |

### Why POST not GET?
> "POST is correct because we're creating a new prediction resource. GET should be idempotent and cacheable, but sentiment predictions are dynamic operations on submitted data."

### Performance Optimization
> "The model loads once at Django startup, not per request. This is critical - loading a model takes ~500ms, but inference is <1ms. We use module-level initialization."

```python
# Loaded ONCE at startup
classifier = SentimentClassifier()
classifier.load(settings.MODEL_DIR)

# Used per request (fast)
result = classifier.evaluate(sentence)
```

---

## 7. EDGE CASES HANDLED

| Edge Case | Response |
|-----------|----------|
| Empty string | 400 Bad Request |
| Missing `sentence` field | 400 Bad Request |
| Text > 5000 chars | 400 Bad Request |
| Model not loaded | 503 Service Unavailable |
| Empty after preprocessing | `{"sentiment": "neutral", "confidence": 0.0, "error": "..."}` |
| Emojis only | Returns neutral (stripped by preprocessing) |
| Invalid JSON | 400 Bad Request |
| Wrong HTTP method | 405 Method Not Allowed |

---

## 8. CONFIDENCE vs PROBABILITY

### Probability
- Always P(positive) from 0 to 1
- Raw model output

### Confidence
- Confidence in the PREDICTED class
- Varies by prediction:

| Prediction | Confidence Calculation |
|------------|----------------------|
| Positive | `probability` |
| Negative | `1 - probability` |
| Neutral | `1 - abs(probability - 0.5) * 2` |

### Example
```
"I hate this" → prob=0.12, sentiment=negative, confidence=0.88
```
We report confidence=0.88 (not 0.12) because we're 88% sure it's negative.

---

## 9. KEY METRICS TO REMEMBER

| Metric | Value | What It Means |
|--------|-------|---------------|
| Binary Accuracy | ~80% | Pos vs Neg classification |
| 3-Class Accuracy | ~60% | Including neutral predictions |
| Training Samples | 1,596,303 | After removing empty |
| Vocabulary Size | 10,000 | TF-IDF features |
| Model Size | ~80 KB | Logistic Regression weights |
| Vectorizer Size | ~360 KB | TF-IDF vocabulary |
| Inference Time | <1ms | Per prediction |

---

## 10. POTENTIAL INTERVIEW QUESTIONS

### Q: Why not use a pre-trained model like BERT?
> "BERT would give ~5-10% higher accuracy but adds significant complexity. For a synchronous API, inference time matters. Logistic Regression gives sub-millisecond inference vs 100-500ms for BERT. The requirements emphasized clean, maintainable code over state-of-the-art accuracy."

### Q: How would you improve the model?
> "Three approaches:
> 1. **More data**: Add neutral-labeled training data
> 2. **Better features**: Add sentiment lexicons (VADER, AFINN)
> 3. **Ensemble**: Combine with a deep learning model for difficult cases"

### Q: Why didn't you use word embeddings (Word2Vec, GloVe)?
> "TF-IDF with bigrams captures sentiment patterns effectively for tweets. Word embeddings are better for semantic similarity but don't inherently capture sentiment. Also, TF-IDF is simpler to deploy and debug."

### Q: How does the model handle sarcasm?
> "Honestly, it doesn't handle sarcasm well. 'Oh great, another Monday' would likely be classified as positive. Detecting sarcasm requires deeper understanding - either a more complex model (BERT) or sarcasm-specific training data."

### Q: What if you had 100x more data?
> "I'd consider:
> 1. Still use Logistic Regression first (scales well)
> 2. Add regularization tuning (C parameter)
> 3. Consider deep learning only if LR accuracy plateaus
> 4. Implement batch processing for training"

### Q: How would you deploy this to production?
> "1. Containerize with Docker
> 2. Add health check endpoint
> 3. Implement request logging and monitoring
> 4. Add rate limiting
> 5. Consider async processing for batch requests
> 6. Load model via gunicorn/uvicorn workers"

### Q: Why Django REST Framework?
> "DRF is the standard for Django APIs. It provides:
> - Serializers for validation
> - Proper HTTP status code handling
> - Browsable API for debugging
> - The requirements mentioned Avoma uses DRF"

### Q: How do you handle model updates?
> "Current setup: restart server to reload model. Production improvement: implement model versioning with A/B testing capability, or use a model registry."

---

## 11. CODE STRUCTURE (Know This!)

```
AVOMA project/
├── sentiment_analysis_logistic_regression.ipynb  # Main notebook (EDA + training)
├── sentiment_model/
│   ├── classifier.py      # SentimentClassifier class
│   └── train_model.py     # CLI training script
├── models/
│   ├── model.joblib       # Saved LogisticRegression
│   └── vectorizer.joblib  # Saved TF-IDF vectorizer
├── api/
│   ├── sentiment/
│   │   ├── views.py       # POST /evaluate endpoint
│   │   └── serializers.py # Request/response validation
│   └── sentiment_api/
│       └── settings.py    # Django config
├── requirements.txt
└── README.md
```

### Why Two Places for Training?
- **Notebook**: For exploration, EDA, visualization, documentation
- **classifier.py**: Clean Python module for API to import
- **Both produce identical results** (same preprocessing, same model params)

---

## 12. QUICK FACTS TO MEMORIZE

- Dataset: **Sentiment140** (1.6M tweets)
- Encoding: **TF-IDF with bigrams**
- Model: **Logistic Regression**
- Framework: **Django + DRF**
- Binary Accuracy: **~80%**
- Neutral Strategy: **Probability thresholds (0.4-0.6)**
- Why no stopwords: **Preserves negation ("not good")**
- Why not BERT: **Inference speed, deployment simplicity**
- Model loads: **Once at startup, not per request**

---

## 13. THINGS TO SAY CONFIDENTLY

- "I chose Logistic Regression for its balance of accuracy, interpretability, and deployment simplicity."
- "Bigrams are essential for sentiment - they capture phrases like 'not good' and 'can't wait'."
- "The low neutral accuracy is expected since we have no neutral training data. We use model uncertainty as a proxy."
- "Loading the model once at startup is critical for API performance."
- "I tested multiple approaches - SVM, Naive Bayes, Random Forest - and they all performed similarly on this dataset."

---

Good luck with your interview! 🎯
