# Crop Recommendation System using Machine Learning

A precision agriculture decision support web application built with **Python**, **Flask**, and **Scikit-Learn** using a **Random Forest Classifier** trained on soil nutrients and climatic parameters.

---

## 🚀 Live Demo & Run Locally
The project is configured for easy local hosting and cloud deployment.

### Local Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mandalejayesh10-stack/CropRecommendation.git
   cd CropRecommendation
   ```

2. **Set up a Virtual Environment (Optional but recommended)**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model (Optional)**
   *Note: Pre-trained serialization files (`model.pkl`, `scaler.pkl`, `label_encoder.pkl`) are already included in the repository. If you wish to retrain the model on the Kaggle dataset, run:*
   ```bash
   python train_model.py
   ```

5. **Run the Flask Web App**
   ```bash
   python app.py
   ```
   Open your browser and navigate to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

---

## 🛠️ Tech Stack
*   **Frontend**: HTML5, CSS3, Bootstrap 5, FontAwesome Icons.
*   **Backend**: Python, Flask (Web framework).
*   **Machine Learning**: Scikit-Learn, Pandas, NumPy.
*   **Visualization**: Matplotlib, Seaborn.
*   **Production Server**: Gunicorn.

---

## 📊 Dataset & Features
The model is trained on the **Kaggle Crop Recommendation Dataset** (2,200 rows of agricultural samples).

### Input Features:
1.  **Nitrogen (N)**: Ratio of Nitrogen content in soil.
2.  **Phosphorus (P)**: Ratio of Phosphorous content in soil.
3.  **Potassium (K)**: Ratio of Potassium content in soil.
4.  **Temperature**: Ambient temperature in °C.
5.  **Humidity**: Relative humidity in %.
6.  **pH**: Soil pH acidity level (0 - 14).
7.  **Rainfall**: Annual rainfall in mm.

### Target:
*   **Label**: Recommended crop (22 unique categories, e.g., *rice, maize, chickpeas, kidney beans, banana, mango, grapes, apple, orange, papaya, coconut, cotton, jute, coffee*).

---

## 📈 Model Performance
The Random Forest Classifier was evaluated using a stratified 80/20 train-test split:
*   **Accuracy**: **99.55%**
*   **Precision**: **99.57%**
*   **Recall**: **99.55%**
*   **F1 Score**: **99.55%**

*All training, validation charts (e.g., correlation heatmap, boxplot, histograms, and confusion matrix) are generated dynamically and saved inside the `static/` directory.*

---

## 📂 Project Structure
```
CropRecommendation/
│
├── app.py                   # Flask server application
├── train_model.py           # Machine learning training & EDA exporter
├── generate_charts.py       # Helper script to export specific static charts
├── requirements.txt         # Package dependencies
├── .gitignore               # Excludes build and workspace files
│
├── dataset/
│   └── Crop_recommendation.csv   # Local agricultural dataset
│
├── templates/
│   ├── index.html           # Main dashboard form input UI
│   └── result.html          # Prediction results display
│
├── static/
│   ├── style.css            # Custom premium agronomy layout stylesheet
│   ├── crop.jpg             # High-quality agricultural banner image
│   ├── heatmap.png          # Correlation matrix
│   ├── crop_distribution.png
│   ├── rainfall_histogram.png
│   ├── ph_histogram.png
│   ├── accuracy_comparison.png
│   ├── boxplot.png
│   └── confusion_matrix.png
│
├── report.md                # Comprehensive academic mini-project report
└── walkthrough.md           # Deployment overview & validation logs
```

---

## 🌐 Cloud Deployment (Render / Railway)
This project is configured with a production WSGI entry point for services like **Render** or **Railway**.

### Steps for deploying on Render:
1. Create a free account on **[Render](https://render.com)** and connect your GitHub profile.
2. Click **New +** and select **Web Service**.
3. Select this repository: `mandalejayesh10-stack/CropRecommendation`.
4. Configure the settings:
   *   **Runtime**: `Python`
   *   **Build Command**: `pip install -r requirements.txt`
   *   **Start Command**: `gunicorn app:app`
5. Click **Deploy Web Service**. Render will dynamically bind the server to the needed `$PORT` variable and host it publicly.
