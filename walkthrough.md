# Project Walkthrough - Crop Recommendation System

The final-year college mini-project has been successfully built and verified. The application is running locally on `http://127.0.0.1:5000`.

---

## 1. Accomplishments & Files Created

All files are organized in the workspace folder: `c:/Users/JAYESH/Documents/aditya project/CropRecommendation/`

- **[requirements.txt](file:///c:/Users/JAYESH/Documents/aditya%20project/CropRecommendation/requirements.txt)**: List of dependencies (`flask`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).
- **[train_model.py](file:///c:/Users/JAYESH/Documents/aditya%20project/CropRecommendation/train_model.py)**: Python training script that:
  - Downloads `Crop_recommendation.csv` automatically if not present.
  - Performs data inspection and cleaning (0 nulls, 0 duplicates found).
  - Saves preprocessors: `label_encoder.pkl` and `scaler.pkl` ($StandardScaler$).
  - Trains a Random Forest Classifier with **99.55% accuracy** on testing data.
  - Saves the model to `model.pkl`.
  - Saves high-quality EDA and validation plots as png files under `static/images/`.
- **[app.py](file:///c:/Users/JAYESH/Documents/aditya%20project/CropRecommendation/app.py)**: Flask backend server routing:
  - `GET /`: renders the homepage with forms.
  - `POST /predict`: takes soil & environment parameters, standardizes them, feeds them into the Random Forest model, decodes the crop label, and returns the result.
- **[static/style.css](file:///c:/Users/JAYESH/Documents/aditya%20project/CropRecommendation/static/style.css)**: A premium forest green/sage style agronomy-themed UI layout.
- **[static/crop.jpg](file:///c:/Users/JAYESH/Documents/aditya%20project/CropRecommendation/static/crop.jpg)**: High-resolution custom generated agricultural landscape image.
- **[templates/index.html](file:///c:/Users/JAYESH/Documents/aditya%20project/CropRecommendation/templates/index.html)**: Main form containing input fields for N, P, K, pH, temperature, humidity, and rainfall, with tooltips and client-side Bootstrap form validation.
- **[templates/result.html](file:///c:/Users/JAYESH/Documents/aditya%20project/CropRecommendation/templates/result.html)**: Recommendation display screen with a detailed summary table showing the submitted variables.
- **[report.md](file:///c:/Users/JAYESH/Documents/aditya%20project/CropRecommendation/report.md)**: A complete, academic project report adhering to the requested college layout, ready for submission.

---

## 2. Model Performance Summary

The Random Forest Classifier was evaluated on a 20% stratified test split and achieved the following metrics:
- **Accuracy**: 99.55%
- **Precision**: 99.57%
- **Recall**: 99.55%
- **F1 Score**: 99.55%

### Exported Figures for Project Report
All generated plots are located in the `static/` directory:
1. **Correlation Heatmap**: [heatmap.png](file:///c:/Users/JAYESH/Documents/aditya%20project/CropRecommendation/static/heatmap.png)
2. **Crop Distribution Chart**: [crop_distribution.png](file:///c:/Users/JAYESH/Documents/aditya%20project/CropRecommendation/static/crop_distribution.png)
3. **Rainfall Distribution Histogram**: [rainfall_histogram.png](file:///c:/Users/JAYESH/Documents/aditya%20project/CropRecommendation/static/rainfall_histogram.png)
4. **Soil pH Distribution Histogram**: [ph_histogram.png](file:///c:/Users/JAYESH/Documents/aditya%20project/CropRecommendation/static/ph_histogram.png)
5. **Model Accuracy Comparison**: [accuracy_comparison.png](file:///c:/Users/JAYESH/Documents/aditya%20project/CropRecommendation/static/accuracy_comparison.png)
6. **Outlier Check Boxplot**: [boxplot.png](file:///c:/Users/JAYESH/Documents/aditya%20project/CropRecommendation/static/boxplot.png)
7. **Validation Confusion Matrix**: [confusion_matrix.png](file:///c:/Users/JAYESH/Documents/aditya%20project/CropRecommendation/static/confusion_matrix.png)

---

## 3. Local Web App Verification

The Flask server was successfully started and validated:
- The server is running at: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.
- Making a mock POST request to the prediction endpoint with standard inputs returned a successful crop prediction for **rice** (verified via mock query test).
- To launch or run the app manually at any time, execute:
  ```bash
  python app.py
  ```

---

## 4. GitHub Repository & Deployment
- **GitHub URL**: **[https://github.com/mandalejayesh10-stack/CropRecommendation](https://github.com/mandalejayesh10-stack/CropRecommendation)**
- The repository has been created and the initial commit pushed.
- A `.gitignore` has been configured to exclude build junk while keeping the necessary model pickles (`model.pkl`, `scaler.pkl`, `label_encoder.pkl`) and data CSVs, enabling zero-configuration deployment.
- `requirements.txt` includes `gunicorn` as the web gateway for cloud deployment (e.g. Render, Railway).
- `app.py` has been updated to dynamically bind to the environment port (`PORT`) for seamless cloud hosting.
