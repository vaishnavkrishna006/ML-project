# Movie Recommendation ML API

A clean, production-ready machine learning recommendation system powered by LightFM.

## Features
- True Matrix Factorization using LightFM with WARP loss.
- Flask API for serving real-time recommendations.
- Modular architecture separating models, data loading, and NLP logic.

## Project Structure
```text
ML-project/
├── data/
│   └── sample_data.csv          # Movie ratings dataset (user_id, movie_id, rating)
├── src/
│   ├── models/
│   │   └── lightfm_model.py     # LightFM recommender implementation
│   ├── nlp/
│   │   └── nlp_query_processor.py # NLP intent extraction (decoupled)
│   ├── utils/
│   │   └── data_loader.py       # Data loading utilities
├── app.py                       # Flask API
├── train_model.py               # Script to train and evaluate the model
├── requirements.txt             # Project dependencies
└── README.md
```

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Run the training script to build the model from `sample_data.csv` (or auto-generated dummy data) and save it to `model.pkl`.
```bash
python train_model.py
```

### 3. Start the API Server
Start the Flask server which will load the trained `model.pkl` on startup.
```bash
python app.py
```

### 4. API Usage
- **Health Check**: `GET http://127.0.0.1:5000/`
- **Get Recommendations**: `GET http://127.0.0.1:5000/recommend/<user_id>?top_k=5`

*Example response:*
```json
{
  "user_id": "1",
  "recommendations": [104, 105],
  "top_k": 5
}
```
