import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Load and prepare raw data
data_path = "urdu_5k_dataset.xlsx"
data = pd.read_excel(data_path)

# Handle text columns cleanly
texts = data.iloc[:, 0].fillna("").astype(str)
labels = data.iloc[:, 1]

# Quick sanity check on target distribution
print("Class Distribution:")
print(labels.value_counts())
print("-" * 40)

# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    texts, 
    labels, 
    test_size=0.20, 
    random_state=42, 
    stratify=labels
)

# 3. Text Feature Extraction (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 4. Model Setup & Fitting
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# 5. Model Evaluation
predictions = clf.predict(X_test_vec)
score = accuracy_score(y_test, predictions)

print(f"Validation Accuracy: {score:.4f}\n")
print("Classification Breakdown:")
print(classification_report(y_test, predictions))

# 6. Save Artifacts for Inference
with open("urdu_model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(tfidf, vec_file)

print("Pipeline artifacts successfully saved.")