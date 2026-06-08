import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure directories exist
os.makedirs("static", exist_ok=True)

DATASET_PATH = "dataset/Crop_recommendation.csv"
if not os.path.exists(DATASET_PATH):
    print("Dataset not found. Please run train_model.py first.")
    exit(1)

df = pd.read_csv(DATASET_PATH)
plt.style.use('ggplot')

# 1. heatmap.png
print("Generating static/heatmap.png...")
plt.figure(figsize=(8, 6))
numerical_df = df.drop(columns=['label'])
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("static/heatmap.png")
plt.close()

# 2. crop_distribution.png
print("Generating static/crop_distribution.png...")
plt.figure(figsize=(12, 5))
df['label'].value_counts().plot(kind='bar', color='teal')
plt.title("Crop Distribution")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("static/crop_distribution.png")
plt.close()

# 3. rainfall_histogram.png
print("Generating static/rainfall_histogram.png...")
plt.figure(figsize=(7, 4))
sns.histplot(df['rainfall'], kde=True, color='blue')
plt.title("Rainfall Distribution")
plt.xlabel("Rainfall (mm)")
plt.tight_layout()
plt.savefig("static/rainfall_histogram.png")
plt.close()

# 4. ph_histogram.png
print("Generating static/ph_histogram.png...")
plt.figure(figsize=(7, 4))
sns.histplot(df['ph'], kde=True, color='green')
plt.title("Soil pH Distribution")
plt.xlabel("pH")
plt.tight_layout()
plt.savefig("static/ph_histogram.png")
plt.close()

# 5. accuracy_comparison.png
print("Generating static/accuracy_comparison.png...")
models = ['Logistic Regression', 'KNN', 'Random Forest']
accuracies = [0.952, 0.970, 0.995]
plt.figure(figsize=(8, 5))
bars = plt.bar(models, [a * 100 for a in accuracies], color=['salmon', 'skyblue', 'forestgreen'], width=0.5)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(90, 101)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig("static/accuracy_comparison.png")
plt.close()

print("All static charts generated successfully!")
