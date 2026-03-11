import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("iris_model.pkl")
encoder = joblib.load("lable_encoder.pkl")

# ---------------- HEADER ----------------
st.title("🌸 Iris Flower Species Prediction")

st.markdown("""
### 👩‍💻 Developed by **Tasneem Begum**

This interactive machine learning app predicts the **species of an Iris flower**
based on its **sepal and petal measurements**.

Use the **sidebar controls** to enter the flower measurements and click **Predict**.
""")

st.divider()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🌼 Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

predict_btn = st.sidebar.button("🔍 Predict Species")

# ---------------- INPUT DATA ----------------
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# ---------------- FEATURE GRAPH ----------------
st.subheader("📊 Input Feature Visualization")

feature_df = pd.DataFrame({
    "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
    "Value": [sepal_length, sepal_width, petal_length, petal_width]
})

st.bar_chart(feature_df.set_index("Feature"))

# ---------------- LOCAL FLOWER IMAGES ----------------
iris_images = {
    "setosa": "images/iris_setosa.png",
    "versicolor": "images/Iris_versicolor.jpg",
    "virginica": "images/Iris_virginica.jpg"
}

# ---------------- PREDICTION ----------------
if predict_btn:

    with st.spinner("🌸 Analyzing flower measurements..."):
        prediction = model.predict(input_data)
        species = encoder.inverse_transform(prediction)[0]
        probabilities = model.predict_proba(input_data)[0]

    clean_species = species.lower().replace("iris-", "")

    # Snow animation
    st.snow()

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"🌼 Predicted Species: **{clean_species.capitalize()}**")

    with col2:
        st.image(
            iris_images[clean_species],
            caption=f"Iris {clean_species.capitalize()}",
            width=300
        )

    # ---------------- PROBABILITY GRAPH ----------------
    st.subheader("📈 Prediction Confidence")

    prob_df = pd.DataFrame({
        "Species": encoder.classes_,
        "Probability": probabilities
    })

    st.bar_chart(prob_df.set_index("Species"))

# ---------------- INFORMATION SECTION ----------------
st.divider()

st.markdown("""
### 🌺 Iris Flower Species

The model predicts between three iris species:

• **Setosa**  
• **Versicolor**  
• **Virginica**

### ⚙️ Model Details
- Algorithm: Machine Learning Classifier  
- Dataset: Iris Dataset  
- Deployment: Streamlit  

This project demonstrates how **machine learning models can be turned into interactive web applications**.
""")