# eval_lstm.py
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, hamming_loss, classification_report

# Load test data
print("Loading test data...")
test_df = pd.read_csv("data/test_clean.csv")
labels_df = pd.read_csv("data/test_labels.csv")

# Filter invalid rows
labels_df = labels_df[labels_df['toxic'] != -1].reset_index(drop=True)
test_df = test_df.loc[labels_df.index].reset_index(drop=True)

X_test = test_df['comment_text'].fillna("")
y_true = labels_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

# Load tokenizer and model
tokenizer = joblib.load("backend/tokenizer.joblib")
model = load_model("backend/lstm_model.keras")

# Preprocess
sequences = tokenizer.texts_to_sequences(X_test)
X_pad = pad_sequences(sequences, maxlen=200, padding='post')

# Predict (on sample to avoid OOM)
SAMPLE_SIZE = 3000
X_sample = X_pad[:SAMPLE_SIZE]
y_sample = y_true[:SAMPLE_SIZE]

print("Predicting...")
y_probs = model.predict(X_sample)
y_pred = (y_probs > 0.3).astype(int)  # Match main.py threshold

# Metrics
print("\nF1 Score (macro):", f1_score(y_sample, y_pred, average='macro'))
print("F1 Score (micro):", f1_score(y_sample, y_pred, average='micro'))
print("Hamming Loss:", hamming_loss(y_sample, y_pred))
print("\nDetailed Classification Report:")
print(classification_report(
    y_sample, y_pred,
    target_names=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
))
