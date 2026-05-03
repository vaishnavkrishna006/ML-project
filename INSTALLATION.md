# ⚙️ Installation & Setup Guide

Follow these steps to get **CineMatch** running on your local machine.

## 📋 Prerequisites
- Python 3.9 or higher
- Windows, macOS, or Linux

## 🛠️ Step 1: Virtual Environment
It is highly recommended to use a virtual environment to keep dependencies organized.

```powershell
# Create environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

## 📦 Step 2: Install Dependencies
Install all required libraries including Pandas, Scikit-learn, Flask, and Streamlit.

```bash
pip install -r requirements.txt
```

## 🧠 Step 3: Model Training
Before you can get recommendations, the AI needs to process the movie database.

```bash
python train_model.py
```
*Note: This script merges the classic MovieLens dataset with modern collections (Marvel, DC, etc.) and generates `src/models/artifacts/recommender.pkl`.*

## 🚀 Step 4: Run the Application

### Option A: Interactive Dashboard (Recommended)
This is the best way to experience CineMatch.
```bash
streamlit run dashboard/app.py
```

### Option B: Flask API
If you need to use the system as a backend service.
```bash
python app.py
```

## 🧪 Testing the API
Once the Flask server is running, you can test it using a browser or Postman:
- **Health Check**: `http://localhost:5000/health`
- **Recommend by Title**: `http://localhost:5000/recommend?title=Toy Story (1995)`
- **Natural Language Search**: `http://localhost:5000/recommend?query=I want a sci-fi movie`

---
**Troubleshooting**: If the dashboard shows "Model not found", ensure you ran `python train_model.py` and that the `src/models/artifacts/recommender.pkl` file was created.
