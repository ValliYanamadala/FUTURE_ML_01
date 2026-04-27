# -------------------------------------------
# SUPPORT TICKET CLASSIFICATION PROJECT
# -------------------------------------------

import pandas as pd
import numpy as np
import re
import os
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# Ensure stopwords are available (download only if missing)
try:
    stopwords.words('english')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass

# -------------------------------------------
# Load Dataset
# -------------------------------------------

data = pd.read_csv("customer_support_tickets.csv")

print("Dataset Preview:")
print(data.head())

# -------------------------------------------
# Combine Subject + Description
# -------------------------------------------

data["text"] = data["Ticket Subject"].fillna('') + " " + data["Ticket Description"].fillna('')

# -------------------------------------------
# Text Cleaning
# -------------------------------------------

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

data["clean_text"] = data["text"].apply(clean_text)

# -------------------------------------------
# Feature Extraction
# -------------------------------------------

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data["clean_text"])

# Target variables
y_category = data["Ticket Type"]
y_priority = data["Ticket Priority"]

# -------------------------------------------
# Train Test Split
# -------------------------------------------

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    X, y_category, test_size=0.2, random_state=42
)

X_train_pri, X_test_pri, y_train_pri, y_test_pri = train_test_split(
    X, y_priority, test_size=0.2, random_state=42
)

# -------------------------------------------
# Train Category Model
# -------------------------------------------

category_model = LogisticRegression(max_iter=200)

category_model.fit(X_train_cat, y_train_cat)

pred_cat = category_model.predict(X_test_cat)

# -------------------------------------------
# Train Priority Model
# -------------------------------------------

priority_model = LogisticRegression(max_iter=200)

priority_model.fit(X_train_pri, y_train_pri)

pred_pri = priority_model.predict(X_test_pri)

# -------------------------------------------
# Evaluation - Category
# -------------------------------------------

print("\nCategory Classification Results")
print("----------------------------------")

print("Accuracy:", accuracy_score(y_test_cat, pred_cat))
print(classification_report(y_test_cat, pred_cat))

# -------------------------------------------
# Evaluation - Priority
# -------------------------------------------

print("\nPriority Prediction Results")
print("----------------------------------")

print("Accuracy:", accuracy_score(y_test_pri, pred_pri))
print(classification_report(y_test_pri, pred_pri))

# -------------------------------------------
# Confusion Matrix
# -------------------------------------------

cm = confusion_matrix(y_test_cat, pred_cat)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Ticket Type Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
output_path = os.path.join(os.getcwd(), "ticket_type_confusion_matrix.png")
plt.savefig(output_path, dpi=200)
print(f"\nSaved confusion matrix plot to: {output_path}")

# -------------------------------------------
# Prediction Function
# -------------------------------------------

def predict_ticket(ticket_text):

    cleaned = clean_text(ticket_text)

    vector = vectorizer.transform([cleaned])

    category = category_model.predict(vector)[0]

    priority = priority_model.predict(vector)[0]

    print("\nTicket:", ticket_text)
    print("Predicted Category:", category)
    print("Predicted Priority:", priority)

# -------------------------------------------
# Test Predictions
# -------------------------------------------

predict_ticket("My payment was deducted twice")

predict_ticket("Unable to login to my account")

predict_ticket("How do I reset my password?")
