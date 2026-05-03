# 🎬 CineMatch: AI Movie Recommendation System

CineMatch is a modern, high-performance movie recommendation engine built with Python. It uses **TF-IDF Vectorization** and **Cosine Similarity** to provide intelligent movie suggestions based on titles, genres, and natural language descriptions.

## 🌟 Features
- **Title-Based Recommendations**: Find movies similar to your favorites (e.g., "Inception").
- **Natural Language Search**: Type how you feel (e.g., "I want a dark crime thriller") and get relevant matches.
- **Modern Collections**: Includes classic MovieLens 100k data PLUS modern **Marvel (MCU)**, **DC**, and **Warner Bros** collections.
- **Interactive Dashboard**: A sleek Streamlit-based UI for exploring the movie universe.
- **Lightweight & Fast**: No complex C++ dependencies or heavy neural networks. Runs perfectly on Windows.

## 🏗️ Project Structure
```text
ML-project/
├── data/
│   └── ml-100k/             # Legacy MovieLens dataset
├── src/
│   └── models/
│       ├── artifacts/       # Saved model (recommender.pkl)
│       └── recommender.py   # Core ML logic (TF-IDF + Cosine Similarity)
├── dashboard/
│   └── app.py               # Streamlit UI (CineMatch Dashboard)
├── app.py                   # Flask API for integration
├── train_model.py           # Training pipeline (Data Processing + ML)
├── requirements.txt         # Project dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the AI Model
Run the training script to process the MovieLens data and the Marvel/DC collections.
```bash
python train_model.py
```

### 3. Launch the Dashboard
Experience the recommendations in a beautiful UI.
```bash
streamlit run dashboard/app.py
```

### 4. (Optional) Run API
If you want to integrate CineMatch with other apps:
```bash
python app.py
```

## 📊 How it Works
The system uses **Natural Language Processing (NLP)**:
1.  **Feature Extraction**: It combines movie genres, tags, and keywords into a text "profile".
2.  **Vectorization**: Uses `TfidfVectorizer` to convert these profiles into mathematical vectors.
3.  **Similarity Calculation**: When you search, it calculates the **Cosine Similarity** between your query and all movies in the database to find the perfect match.

## 🍿 Dataset
- **MovieLens 100k**: 1,600+ classic movies.
- **Custom Collections**: Modern hits from Marvel, DC, and WB added manually to ensure relevant results for 2024.

---
Built for academic and experimental use in Machine Learning.
