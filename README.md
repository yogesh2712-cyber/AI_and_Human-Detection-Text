# AI_and_Human-Detection-Text
🧠 AI vs Human Text Detector

A Machine Learning–based web application that detects whether a given text is AI-generated or human-written.
The system uses TF-IDF vectorization and a trained ML classification model, deployed via a Flask web interface.

🚀 Features

🔍 Detects AI-generated vs Human-written text

📊 Shows confidence score (e.g., 92%)

🌐 Simple and clean web UI

⚡ Fast prediction using pre-trained model

🧠 End-to-end ML pipeline (Data → Model → Web App)

🛠️ Tech Stack

Python

Scikit-learn

TF-IDF Vectorizer

Flask

HTML / CSS

Pickle (Model Serialization)

🧪 Machine Learning Pipeline

1. Data Collection

AI-generated text

Human-written text

2. Data Cleaning

Remove null rows

Fix label column

Balance dataset

3. Text Preprocessing

Lowercasing

Tokenization

Stopword removal

4. Feature Extraction

TF-IDF Vectorization

5. Model Training

Supervised classification model

Saved using Pickle

6. Model Deployment

Flask-based web application

📊 Model Output

Prediction: AI Generated 🤖 / Human Written 🧑

Confidence Score: Percentage probability

Accuracy: Evaluated using test dataset
