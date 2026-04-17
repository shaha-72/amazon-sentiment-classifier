import zipfile
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import joblib
joblib.dump(model, "model.pkl")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# Step 1: Read CSV from ZIP
zip_path = r"E:\sentiment analysis\amazon reviews.csv.zip"

with zipfile.ZipFile(zip_path) as z:
    print("Files inside ZIP:", z.namelist())
    with z.open("1429_1.csv") as f:
        df = pd.read_csv(f, encoding="ISO-8859-1", on_bad_lines='skip', low_memory=False)

# Step 2: Clean and filter data
df = df[['reviews.text', 'reviews.rating']].dropna()
df = df[df['reviews.rating'] != 3]  # Remove neutral reviews
df['Sentiment'] = df['reviews.rating'].apply(lambda x: 1 if x > 3 else 0)
df.rename(columns={'reviews.text': 'Review'}, inplace=True)

# Dataset stats — important for resume/README
print(f"\nDataset size after cleaning: {len(df)} reviews")
print(f"Positive: {df['Sentiment'].sum()} | Negative: {len(df) - df['Sentiment'].sum()}")

# Step 3: Text preprocessing
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

df['Cleaned'] = df['Review'].apply(clean_text)

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Cleaned'])
y = df['Sentiment']

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train Logistic Regression with class balancing
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(n_jobs=-1, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)

# Step 7: Predict and evaluate
lr_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, lr_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred, target_names=["Negative", "Positive"]))

# Step 8: Custom Predictions
print("\nSample Predictions:")
custom_reviews = [
    "I love this product, works great!",
    "Terrible experience, very disappointed.",
    "It's okay, not what I expected.",
    "Absolutely amazing quality!",
    "Stopped working after two days, waste of money.",
    "Best purchase I have made this year!"
]

cleaned_custom = [clean_text(review) for review in custom_reviews]
vec_custom = vectorizer.transform(cleaned_custom)
predictions = lr_model.predict(vec_custom)
probabilities = lr_model.predict_proba(vec_custom)

for review, label, prob in zip(custom_reviews, predictions, probabilities):
    sentiment = "Positive ✅" if label == 1 else "Negative ❌"
    confidence = prob[label] * 100
    print(f'• "{review}"\n  → {sentiment} (Confidence: {confidence:.1f}%)\n')

# Step 9: Confusion Matrix plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, lr_pred),
    display_labels=["Negative", "Positive"]
).plot(ax=axes[0])
axes[0].set_title("Logistic Regression Confusion Matrix")

# Step 10: Sentiment distribution bar chart (NEW)
sentiment_counts = df['Sentiment'].value_counts()
axes[1].bar(
    ['Negative', 'Positive'],
    [sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)],
    color=['#e74c3c', '#2ecc71'],
    edgecolor='black'
)
axes[1].set_title("Sentiment Distribution in Dataset")
axes[1].set_ylabel("Number of Reviews")
for i, v in enumerate([sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)]):
    axes[1].text(i, v + 50, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
plt.close()
