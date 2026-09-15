# SMS & Social Media Spam Classifier

An end-to-end Machine Learning pipeline and Natural Language Processing (NLP) system designed to detect and classify spam messages and social media comments using Multinomial Naive Bayes and TF-IDF Vectorization.

## 📌 Features
* **Text Preprocessing**: Automated cleaning, stop-word removal, and tokenization using `NLTK`.
* **Feature Extraction**: Term Frequency-Inverse Document Frequency (`TF-IDF`) vectorization.
* **Machine Learning Model**: `MultinomialNB` classifier serialized via `joblib`.
* **Cross-Dataset Validation**: Benchmarked against standard public social media spam collections (YouTube Spam Dataset).

## 📁 Repository Structure
├── sms_spam_classifier.ipynb   # Main Google Colab notebook containing training & evaluation
├── spam_model.pkl              # Serialized MultinomialNB model
├── tfidf_vectorizer.pkl        # Serialized TF-IDF vectorizer
└── README.md                   # Project documentation

## 🛠️ Installation & Setup

1. **Clone the repository:**
   git clone [https://github.com/code-with-sarah/SMS-Spam-Classifier.git](https://github.com/code-with-sarah/SMS-Spam-Classifier.git)
   cd SMS-Spam-Classifier

2. **Install required dependencies:**
pip install pandas numpy scikit-learn nltk joblib

3. **Run Inference in Python:**
   import joblib

# Load saved assets
model = joblib.load('spam_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Predict single sample
sample_text = ["Congratulations! You've won a free ticket. Call now!"]
vectorized_text = tfidf.transform(sample_text).toarray()
prediction = model.predict(vectorized_text)[0]

print("SPAM" if prediction == 1 else "HAM")

📊 Evaluation & Results
The model achieves strong accuracy and precision metrics when evaluated on benchmark public message and comment spam datasets:

Model: Multinomial Naive Bayes

Vectorizer: TF-IDF Feature Extractor
