# Toxic Comment Classification (LSTM-based Web App)

This project is a full-stack machine learning web application that classifies user input text into multiple toxic comment categories using an LSTM neural network.

## 🔍 Features
- Multi-label classification with 6 toxic categories:
  - `toxic`
  - `severe_toxic`
  - `obscene`
  - `threat`
  - `insult`
  - `identity_hate`
- Deep learning model trained on Kaggle's Jigsaw dataset
- Frontend built with HTML + CSS
- Backend using FastAPI
- Custom confidence threshold for filtering predictions

---

## 🚀 Getting Started

### 1. Clone the repository
```
git clone <your-repo-url>
cd toxic_classifier
```

### 2. Create and activate virtual environment (Python 3.10 recommended)
```
python3.10 -m venv venv
source venv/bin/activate  # or source venv/bin/activate.fish
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Train the model (optional)
If `backend/lstm_model.keras` and `backend/tokenizer.joblib` are not available:
```
python train_lstm.py
```

### 5. Run the web app locally
```
uvicorn backend.main:app --reload
```
Then visit: [http://localhost:8000](http://localhost:8000)

---

## 📁 Project Structure
```
.
├── backend/
│   ├── main.py               # FastAPI backend
│   ├── lstm_model.keras      # Trained LSTM model
│   ├── tokenizer.joblib      # Tokenizer object
├── data/
│   ├── train_clean.csv       # Preprocessed training data
│   ├── test_clean.csv        # Preprocessed test data
│   ├── test_labels.csv       # True labels for evaluation
├── frontend/
│   ├── index.html            # Frontend form
├── train_lstm.py             # LSTM training script
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🧪 Example Inputs
Try pasting comments like:
```
You're such a disgusting idiot. No one needs your opinion.
```
Expected output:
```
toxic, insult, obscene
```

---

## 📜 License
This project is for educational purposes only.

---

## 👤 Author
Maral Abay — 2025 final project for Machine Learning


