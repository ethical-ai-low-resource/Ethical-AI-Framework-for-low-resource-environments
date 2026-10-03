import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def train_pipeline(data_path="urdu_5k_dataset.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: {data_path} file nahi mili! File name check karein.")

    # 1. Load Data cleanly with UTF-8 support
    print(f"[INFO] Loading dataset from: {data_path}")
    data = pd.read_csv(data_path, encoding='utf-8-sig')

    # Ensure non-empty dataframe
    if data.shape[1] < 2:
        raise ValueError("Error: Dataset mein kam az kam 2 columns (Text, Label) hone chahiye.")

    # 2. Extract Text and Label columns safely
    X_raw = data.iloc[:, 0].fillna("").astype(str)
    y_raw = data.iloc[:, 1].astype(str)

    print("\n--- Target Class Distribution ---")
    print(y_raw.value_counts())
    print("-" * 35)

    # 3. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    # 4. Feature Extraction (TF-IDF Vectorizer)
    # max_features set karne se web deployment mein speed tez ho jati hai
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Model Training (Logistic Regression)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    # 6. Evaluation for Research Paper Reporting
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- Model Performance Metrics ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 35)

    # 7. Save Model Artifacts for Web App Deployment
    with open("urdu_model.pkl", "wb") as f_model:
        pickle.dump(model, f_model)

    with open("vectorizer.pkl", "wb") as f_vec:
        pickle.dump(vectorizer, f_vec)

    print("\n[SUCCESS] 'urdu_model.pkl' aur 'vectorizer.pkl' successfully save ho gaye hain!")

if __name__ == "__main__":
    train_pipeline()