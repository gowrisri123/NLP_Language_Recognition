import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = pd.read_csv("dataset.csv", encoding="utf-8-sig")
data = data.dropna(subset=["text", "language"])

data["text"] = data["text"].astype(str).str.strip()
data["language"] = data["language"].astype(str).str.strip()

# 2. Remove empty and duplicate sentences
data = data[data["text"] != ""]
data = data.drop_duplicates(subset=["text"])

print("Total unique sentences:", len(data))
print("\nSentences per language:")
print(data["language"].value_counts())

# 3. Validate dataset
if data["language"].nunique() < 2:
    raise ValueError("Dataset must contain at least two languages.")

if data["language"].value_counts().min() < 2:
    raise ValueError("Each language needs at least two unique sentences.")

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["language"],
    test_size=0.30,
    random_state=42,
    stratify=data["language"]
)

# 5. Build the machine learning model
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 4)
        )
    ),
    ("classifier", LinearSVC())
])

# 6. Train the model
model.fit(X_train, y_train)

# 7. Evaluate using the held-out test set
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")
print("Training sentences:", len(X_train))
print("Testing sentences:", len(X_test))
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# 8. Save trained model and test accuracy
joblib.dump(model, "language_model.joblib")

with open("model_accuracy.txt", "w", encoding="utf-8") as file:
    file.write(f"{accuracy * 100:.2f}")

print("\nModel saved as language_model.joblib")
print("Test accuracy saved as model_accuracy.txt")