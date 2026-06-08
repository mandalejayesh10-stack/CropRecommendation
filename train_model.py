import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Create directories if they do not exist
os.makedirs("dataset", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("static/images", exist_ok=True)

DATASET_PATH = "dataset/Crop_recommendation.csv"
DATASET_URL = "https://raw.githubusercontent.com/arzzahid66/Optimizing_Agricultural_Production/master/Crop_recommendation.csv"

# Step 1: Download dataset if not present
if not os.path.exists(DATASET_PATH):
    print("Dataset not found locally. Downloading from raw GitHub source...")
    try:
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
        print("Dataset downloaded successfully.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        # Fallback dataset creation if internet fails
        print("Creating a synthetic sample dataset as fallback...")
        # (This is just a fallback, the download should normally succeed)
        synthetic_data = pd.DataFrame({
            'N': np.random.randint(0, 140, 100),
            'P': np.random.randint(5, 145, 100),
            'K': np.random.randint(5, 205, 100),
            'temperature': np.random.uniform(8.0, 43.0, 100),
            'humidity': np.random.uniform(14.0, 100.0, 100),
            'ph': np.random.uniform(3.5, 9.9, 100),
            'rainfall': np.random.uniform(20.0, 300.0, 100),
            'label': np.random.choice(['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'], 100)
        })
        synthetic_data.to_csv(DATASET_PATH, index=False)

# Step 2: Load dataset
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Step 3: Dataset Inspection
print("\n--- DATASET INSPECTION ---")
print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values check:")
missing_vals = df.isnull().sum()
print(missing_vals)
print(f"Total missing values: {missing_vals.sum()}")

print("\nDuplicate values check:")
duplicates_count = df.duplicated().sum()
print(f"Total duplicate rows: {duplicates_count}")
if duplicates_count > 0:
    print("Removing duplicates...")
    df = df.drop_duplicates()
    print(f"New dataset shape: {df.shape}")

# Step 4: Exploratory Data Analysis (EDA)
print("\nGenerating EDA Plots...")
plt.style.use('ggplot')

# 1. Correlation Heatmap (only for numerical columns)
plt.figure(figsize=(10, 8))
numerical_df = df.drop(columns=['label'])
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Soil & Climate Features")
plt.tight_layout()
plt.savefig("static/images/correlation_heatmap.png")
plt.close()
print("Saved correlation_heatmap.png")

# 2. Crop Distribution Bar Chart
plt.figure(figsize=(12, 6))
df['label'].value_counts().plot(kind='bar', color='teal')
plt.title("Distribution of Crop Classes in Dataset")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("static/images/crop_distribution.png")
plt.close()
print("Saved crop_distribution.png")

# 3. Histogram of environmental features
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()
features = numerical_df.columns
for i, feature in enumerate(features):
    sns.histplot(df[feature], ax=axes[i], kde=True, color='forestgreen')
    axes[i].set_title(f"Distribution of {feature}")
# Hide empty subplots if any (7 features, so index 7 and 8 are unused)
for j in range(len(features), 9):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig("static/images/histogram.png")
plt.close()
print("Saved histogram.png")

# 4. Boxplot for Outliers Check
plt.figure(figsize=(14, 8))
sns.boxplot(data=numerical_df, palette='Set2')
plt.title("Boxplot of Soil and Environmental Parameters (Outlier Check)")
plt.tight_layout()
plt.savefig("static/images/boxplot.png")
plt.close()
print("Saved boxplot.png")

# Step 5: Data Preprocessing & Encoding
print("\nPreprocessing Data...")
X = df.drop('label', axis=1)
y = df['label']

# Encode targets
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("Saved label_encoder.pkl")

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Step 7: Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Saved scaler.pkl")

# Step 8: Train Model (Random Forest Classifier)
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("Saved model.pkl")

# Step 9: Model Evaluation
print("\n--- MODEL EVALUATION ---")
y_pred = rf_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
# For multiclass, we use macro or weighted average for precision, recall, f1
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score:  {f1:.4f} ({f1 * 100:.2f}%)")

# Save Confusion Matrix plot
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix - Random Forest Classifier")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("static/images/confusion_matrix.png")
plt.close()
print("Saved confusion_matrix.png")

print("\nModel training and validation complete!")
