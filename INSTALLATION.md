# Installation and Setup Guide

This project uses **LightFM** for movie recommendations. Due to compatibility issues with Python 3.12+ and Windows, we use a community-maintained fork: `lightfm-next`.

## 🛠️ Step 1: Environment Setup

1. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```

2. Activate it:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. Upgrade pip and install build tools:
   ```powershell
   python -m pip install --upgrade pip wheel setuptools
   ```

## 📦 Step 2: Install Dependencies

Install required packages in order:
```powershell
pip install numpy scipy scikit-learn pandas flask streamlit plotly joblib requests
pip install lightfm-next
```

> [!NOTE]
> We use `lightfm-next` because the original `lightfm` currently fails to build on Windows with Python 3.13.

## 🚀 Step 3: Train the Model

Before running the app, you must generate the model artifacts:
```powershell
python train_model.py
```
This will download the MovieLens dataset and save the trained model to `src/models/artifacts/`.

## 📊 Step 4: Run the Project

- **Backend API**: `python app.py`
- **Dashboard**: `streamlit run dashboard/app.py`

---

## 💡 Alternatives for Windows
If you encounter performance issues with LightFM on Windows (e.g., lack of OpenMP support), consider switching to:
1. **Implicit**: Extremely fast matrix factorization with great Windows support.
2. **Surprise**: Easy-to-use scikit-style recommendation library.
