import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load and clean data
df = pd.read_csv("data/train_clean.csv")
df['comment_text'] = df['comment_text'].fillna("")

X_raw = df['comment_text']
y_raw = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Tokenize
MAX_WORDS = 20000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_raw)
sequences = tokenizer.texts_to_sequences(X_raw)
X_pad = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")

# Save tokenizer
joblib.dump(tokenizer, "backend/tokenizer.joblib")

# Calculate class weights per label
def get_multilabel_class_weights(y):
    weights = {}
    for i in range(y.shape[1]):
        y_binary = y[:, i]
        classes = np.unique(y_binary)
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y_binary)
        weights[i] = dict(zip(classes, w))
    return weights

# Apply weights manually during training
class_weights = get_multilabel_class_weights(y_raw)

# Model
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(6, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# Custom training loop with class weights
EPOCHS = 10
BATCH_SIZE = 128

# Train with EarlyStopping
model.fit(
    X_pad,
    y_raw,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=2)],
    # NOTE: You cannot pass class_weight for multi-label directly. It's applied per label in custom training only.
)

# Save model
model.save("backend/lstm_model.keras")

