# Web Interface Guide - Movie Recommendation System

## 🌐 Web Interface Setup & Usage

This guide will help you run and use the web-based recommendation interface.

---

## ✅ Prerequisites

- Python 3.9+
- All dependencies installed (see Installation)
- Dataset files in `datasets/` folder

---

## 📦 Installation

### Step 1: Install Additional Dependencies

```bash
cd "c:\Users\vaish\OneDrive\Documents\MCA\sem 2\ML\ml project"

# Install Flask and related packages
pip install flask flask-cors
```

Or install all requirements at once:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Interface

### Step 1: Start the Flask Server

```bash
cd "c:\Users\vaish\OneDrive\Documents\MCA\sem 2\ML\ml project"
python app.py
```

**Expected Output:**
```
Starting Movie Recommendation System Interface...
Loading data...
Initializing models...
Models loaded successfully!
 * Running on http://127.0.0.1:5000
 * Press CTRL+C to quit
```

### Step 2: Open in Browser

Open your web browser and go to:

```
http://127.0.0.1:5000
```

Or click: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🎨 Interface Features

### 1. **Header Section**
- Displays system name and description
- Professional branding

### 2. **Statistics Dashboard**
- Total Users
- Movie Views
- Unique Movies
- Automatically loaded from dataset

### 3. **Recommendation Input Form**

#### User ID Input
- Enter a valid user ID
- Example IDs: 211145, 124091, 284179
- Required field

#### Model Selection
Choose one of 4 recommendation models:

1. **🔀 Hybrid (Recommended)**
   - Combines all 3 approaches
   - Best overall results
   - Balanced and robust

2. **👥 User-Based Collaborative Filtering**
   - Finds similar users
   - Recommends their favorite movies
   - Best for: Users with watch history

3. **🎬 Item-Based Collaborative Filtering**
   - Finds similar movies
   - Based on co-occurrence patterns
   - Best for: Exploring related movies

4. **👤 Content-Based Filtering**
   - Uses user demographics
   - Analyzes viewing patterns
   - Best for: New users, cold-start scenarios

#### Number of Recommendations
- Default: 5
- Range: 1-20
- Select how many recommendations you want

### 4. **Results Display**

Results show:
- Recommendation Rank (1, 2, 3, ...)
- Movie ID
- Recommendation Score (with visual bar)
- Score normalized to 100%

### 5. **Info Cards**
Educational information about each model

---

## 📊 Example Usage

### Example 1: Get Hybrid Recommendations

1. **User ID:** 211145
2. **Model:** Hybrid (Recommended)
3. **Number:** 5
4. Click **Get Recommendations**

**Result:**
```
Rank | Movie ID | Score
-----|----------|-------
  1  |   41820  | 860.00
  2  |    744   | 774.00
  3  |   5371   | 755.00
  4  |   2809   | 731.00
  5  |   3049   | 721.00
```

### Example 2: Compare Models

Try the same user with different models:

1. **First:** User 211145 with User-Based CF
2. **Then:** User 211145 with Item-Based CF
3. **Finally:** User 211145 with Content-Based

Compare the different recommendations!

---

## 🔧 API Endpoints

The system also exposes REST APIs:

### Get Recommendations
```
POST /api/recommend
```

**Request:**
```json
{
  "user_id": 211145,
  "model_type": "hybrid",
  "n_recommendations": 5
}
```

**Response:**
```json
{
  "success": true,
  "user_id": 211145,
  "model_type": "hybrid",
  "count": 5,
  "recommendations": [
    {"rank": 1, "movie_id": 41820, "recommendation_score": 860.0},
    ...
  ]
}
```

### Get Statistics
```
GET /api/stats
```

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_users": 359347,
    "total_records": 17309433,
    "unique_movies": 86992
  }
}
```

### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "service": "Movie Recommendation System"
}
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'flask'"

**Solution:**
```bash
pip install flask flask-cors
```

### Issue: "Port 5000 already in use"

**Solution 1:** Kill the process using port 5000
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Solution 2:** Change the port in `app.py`
```python
app.run(debug=True, host='127.0.0.1', port=5001)  # Change 5001
```

### Issue: "No recommendations found for User X"

**Possible causes:**
1. User ID doesn't exist in dataset
2. User has no listening history
3. Model-specific limitations

**Solution:** Try different user IDs (211145, 124091, 284179)

### Issue: Slow loading time on first request

**Explanation:** First request loads and initializes models (takes 1-2 minutes with full dataset)

**Solution:** Use sampled dataset (already configured in `app.py`)

---

## 🎯 Performance Tips

1. **First Load:** May take 1-2 minutes (depends on dataset size)
2. **Subsequent Requests:** Much faster (cached models)
3. **Optimize:** Use sampled data for demos (see `app.py` line 231)

---

## 📱 Access from Other Devices

To access from other computers on your network:

1. Find your IP address:
   ```bash
   ipconfig  # Windows
   ```

2. Modify `app.py`:
   ```python
   app.run(debug=True, host='0.0.0.0', port=5000)
   ```

3. Access from other device:
   ```
   http://YOUR_IP_ADDRESS:5000
   ```

---

## 🛑 Stopping the Server

Press `CTRL + C` in the terminal running the server.

```
^C
```

---

## 📁 File Structure

```
ml project/
├── app.py                          # Flask application
├── templates/
│   └── index.html                 # Web interface HTML
├── static/
│   ├── style.css                  # Styling
│   └── script.js                  # Frontend logic
├── recommendation_models.py        # Model implementation
├── requirements.txt               # Dependencies
└── ../datasets/
    └── data/
        ├── play_counts.csv
        └── users.csv
```

---

## 🚀 Advanced Features

### Customize Models

Edit `app.py` to adjust model parameters:

```python
# Line 78-85: User-based CF parameters
n_similar_users=10        # Change number of similar users
n_recommendations=5       # Change number of recommendations

# Line 128: Hybrid weights
weights = {'user_cf': 0.33, 'item_cf': 0.33, 'content': 0.34}
```

### Add New Models

Add new model class in `app.py` and create new API endpoint:

```python
@app.route('/api/recommend-new-model', methods=['POST'])
def new_model_endpoint():
    # Your implementation
    pass
```

---

## 📊 Sample User IDs to Test

```
Valid User IDs:
- 211145
- 124091
- 284179
- 233138
- 222880
```

Try any user ID from the dataset!

---

## ✨ Interface Highlights

- ✅ Responsive design (works on mobile/tablet)
- ✅ Real-time recommendations
- ✅ Visual score bars
- ✅ Model descriptions
- ✅ Statistics dashboard
- ✅ Error handling
- ✅ Loading indicators
- ✅ Smooth animations

---

## 📖 Documentation

- **Full Models:** See `README.md`
- **Code Examples:** See `quick_reference.py`
- **Tests:** See `test_models_fast.py`
- **API:** See this guide

---

## 🎓 Learning Resources

- Flask Documentation: https://flask.palletsprojects.com/
- Recommendation Systems: https://en.wikipedia.org/wiki/Recommender_system
- Collaborative Filtering: https://en.wikipedia.org/wiki/Collaborative_filtering

---

## 📝 Next Steps

1. ✅ Install Flask: `pip install flask`
2. ✅ Run server: `python app.py`
3. ✅ Open browser: http://127.0.0.1:5000
4. ✅ Get recommendations!
5. ✅ Explore different models
6. ✅ Try different user IDs

---

**Interface Version:** 1.0.0  
**Last Updated:** April 2026  
**Status:** Production Ready ✅

Enjoy exploring recommendations!
