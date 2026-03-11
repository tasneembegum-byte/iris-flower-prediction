# 🌸 Iris Flower Species Prediction

🔗 **Live App:**  
https://iris-flower-prediction-3fhtrjrtyjke6bcnx9gvhs.streamlit.app/

---

## 📌 Project Overview

The **Iris Flower Prediction App** is a machine learning web application built using **Python and Streamlit**.  
It predicts the **species of an Iris flower** based on four flower measurements:

- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

The model classifies the flower into one of three species:

- **Setosa**
- **Versicolor**
- **Virginica**

---

## 🚀 Features

- Interactive **Streamlit Web App**
- Sidebar input sliders for flower measurements
- Real-time **species prediction**
- **Prediction confidence visualization**
- **Feature visualization chart**
- Flower **image display based on prediction**
- Snow animation for interactive UI

---

## 🧠 Machine Learning Model

The model predicts the species using four numerical features:

| Feature | Description |
|-------|-------------|
| Sepal Length | Length of the sepal in cm |
| Sepal Width | Width of the sepal in cm |
| Petal Length | Length of the petal in cm |
| Petal Width | Width of the petal in cm |

These measurements allow the model to classify flowers into **three species** of iris plants.

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Scikit-learn  
- Pandas  
- NumPy  
- Joblib  

---

## 📂 Project Structure

```
iris-flower-prediction
│
├── app.py
├── iris_model.pkl
├── lable_encoder.pkl
├── requirements.txt
│
└── images
    ├── iris_setosa.png
    ├── Iris_versicolor.jpg
    └── Iris_virginica.jpg
```


## 📊 Model Workflow

1. Load trained ML model using **joblib**
2. User inputs flower measurements
3. Model predicts iris species
4. Display prediction with confidence chart
5. Show corresponding flower image


---

## 👩‍💻 Author

**Tasneem Begum**

---

## ⭐ Future Improvements

- Add dataset visualization (scatter plots)
- Add model accuracy metrics
- Add model comparison
- Improve UI with more interactive charts
