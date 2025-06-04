import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
target_names = wine.target_names

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Page title
st.title("Model Comparision - wine edition")
st.write("""
Choosing the right machine learning model is a bit like wine tasting -  
you sample a few, swirl them around, and see which one fits best. 
This demo compares four classic models on the Wine dataset.
""")

# Dropdown to select the model
model_choice = st.selectbox("Choose model to test: ", [
    "Random Forest",
    "Logistic Regression",
    "K_Nearest Neighbours",
    "Support Vector Machine"
])

if model_choice == "Random Forest":
    model = RandomForestClassifier(random_state=42)
    st.info("Random Forest handles complexity well and rarely disappoints.")
elif model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=500)
    st.info("Logistic Regression is lean, fast, and often underrated.")
elif model_choice == "K_Nearest Neighbours":
    model = KNeighborsClassifier()
    st.info("KNN is simple and intuitive, but sensitive to noisy features.")
elif model_choice == "Support Vector Machine":
    model = SVC(probability=True)
    st.info(
        "SVM are powerful when decision boundaries are tight&margins matter.")

# Fit
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Accuracy")
st.write(f"The model achieved {acc:.2f} accuracy on the test set.")

# Confusion Matrix
if st.checkbox("Show confusion matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=target_names
    )
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    st.pyplot(fig_cm)

# Optional: Compare all models side by side
if st.checkbox("Compare all models"):
    st.subheader("Model Comparision")
    all_models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "K_Nearest Neighbours": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True)
    }

    results = {}
    for name, clf in all_models.items():
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        results[name] = score

    st.bar_chart(pd.Series(results).sort_values(ascending=False))

# Reflective message
st.markdown("---")
st.markdown("""
Sometimes the best model isn't the fanciest - it's the one that
**fits your story** Try, test, reflect. That's the quiet art of machine learning.

| Data Trait | Try this Model |
|------------|----------------|
| Linear relationships | SVM, Random Forest |
| Nonlinear boundaries | Logistic Regression, RF |
| Interpretable model needed | Logistic Regression, RF |
| High Noise or outliers | Random Forest |
| You want a quick test | KNN or Logistic Regression |
| You're just getting started | Try a few, compare visually |

### Last Note:
Honestly, early in a project, it's not guessing - it's exploring.
Try a few safe bets, compare their behaviour, then follow clues.
Try 3 or 4 models                            
""")
