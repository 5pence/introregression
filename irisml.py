import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML Tools
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()

# Convert to df
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Charting (combine features and species names)
iris_df = X.copy()
iris_df["species"] = pd.Categorical.from_codes(y, iris.target_names)

if st.checkbox("Show mean feature values per species (bar chart)"):
    mean_vals = iris_df.groupby("species").mean()
    st.subheader("Average Feature Values per Species")
    st.bar_chart(mean_vals)

if st.checkbox("Show petal length vs. width (scatter plot)"):
    st.subheader("Petal Lengthvs Width by Species")

    import altair as alt
    chart = alt.Chart(iris_df).mark_circle(size=60).encode(
        x='petal length (cm)',
        y='petal width (cm)',
        color='species',
        tooltip=['species', 'petal length (cm)', 'petal width (cm)']
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train the classic forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# == STREAMLIT APP UI ==
# Set title
st.title("Iris Flower Classifier (Streamlit + ML Demo)")

# Set intro text
st.write("Adjust the sliders below to predict the Iris species")

# Input sliders
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.0)

# Combine the input into a 2D list (needed by sklearn)
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Make a prediction
prediction = clf.predict(input_data)[0]  # Returns 0, 1 or 2
pred_name = iris.target_names[prediction]  # Maps to species name

# Show prediction
st.subheader("Prediction")
st.write(f"The model predicts: {pred_name} Species.")

# Model accuracy (optional checkbox)
if st.checkbox("Show model accuracy"):
    acc = clf.score(X_test, y_test)  # Accuracy on test set
    st.write(f"Model accuracy on test set: {acc:.2f}")

# Confusion Matrix
if st.checkbox("Show confusion matrix"):
    # Predict all X_test
    cm = confusion_matrix(y_test, clf.predict(X_test))

    # Display
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=iris.target_names
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)

# Probability plot
if st.checkbox("Show probability vs petal width"):
    pw_range = np.linspace(0.1, 2.5, 100)
    prob_data = [
        [sepal_length, sepal_width, petal_length, pw] for pw in pw_range]
    probs = clf.predict_proba(prob_data)

    fig2, ax2 = plt.subplots()
    for i, class_name in enumerate(iris.target_names):
        ax2.plot(pw_range, probs[:, i], label=class_name)

    # Label chart
    ax2.set_xlabel("Petal width (cm)")
    ax2.set_ylabel("Prediction Probability")
    ax2.set_title("Prediction Probabilities vs Petal width")
    ax2.legend()
    st.pyplot(fig2)
