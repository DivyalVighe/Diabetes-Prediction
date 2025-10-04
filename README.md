# Diabetes Prediction Dashboard

This is an interactive Streamlit dashboard for predicting diabetes using the Pima Indians Diabetes dataset.  
It allows users to:
- Explore and clean the dataset  
- Perform Exploratory Data Analysis (EDA)  
- Train and compare machine learning models  
- Predict diabetes for custom patient inputs  

---

## Features
- Dataset preview and basic statistics  
- Interactive EDA with filters and visualizations  
- Multiple ML models: Logistic Regression, Decision Tree, Random Forest, and SVM  
- Confusion matrix and ROC curve visualization  
- Real-time prediction using the best-performing model  

---

## Technologies Used
- Python  
- Streamlit  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-dashboard.git
   cd diabetes-dashboard
# Diabetes-Prediction
Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py
Dataset

Uses the Pima Indians Diabetes dataset




# =========================================================
# Diabetes Prediction Dashboard
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# -----------------------------
# Page Config
st.set_page_config(page_title="Diabetes Prediction Dashboard", layout="wide")
st.title("Diabetes Prediction Dashboard")
st.write("Explore dataset, analyze features, train models, and predict diabetes.")

# -----------------------------
# Load Dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
               "BMI","DiabetesPedigreeFunction","Age","Outcome"]
    df = pd.read_csv(url, names=columns)
    zero_invalid = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    df[zero_invalid] = df[zero_invalid].replace(0, np.nan)
    imputer = SimpleImputer(strategy="median")
    df[zero_invalid] = imputer.fit_transform(df[zero_invalid])
    return df

df = load_data()

# -----------------------------
# Fit scaler on full dataset (so it's always available)
X_all = df.drop("Outcome", axis=1)
scaler = StandardScaler()
scaler.fit(X_all)

# Train a default Random Forest model on full dataset for predictions
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(scaler.transform(X_all), df["Outcome"])

# -----------------------------
# Sidebar Navigation
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", ["Dataset Preview", "EDA & Feature Analysis", "Model Training", "Predict Diabetes"])

# -----------------------------
# Dataset Preview
if section == "Dataset Preview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Class Distribution:")
    st.bar_chart(df["Outcome"].value_counts())
    
    if st.sidebar.checkbox("Show Raw Data"):
        st.write(df)

# -----------------------------
# EDA & Feature Analysis
elif section == "EDA & Feature Analysis":
    st.subheader("Exploratory Data Analysis & Feature Analysis")
    
    # Sidebar filters
    st.sidebar.header("Filters for EDA")
    selected_features = st.sidebar.multiselect("Select Features to Compare:", df.columns[:-1],
                                               default=["Glucose","BMI","Age"])
    age_range = st.sidebar.slider("Age Range:", int(df["Age"].min()), int(df["Age"].max()), (20,60))
    preg_range = st.sidebar.slider("Pregnancies Range:", int(df["Pregnancies"].min()), int(df["Pregnancies"].max()), (0,5))
    
    filtered_df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1]) &
                     (df["Pregnancies"] >= preg_range[0]) & (df["Pregnancies"] <= preg_range[1])]
    
    # Line / Bar chart for selected features
    st.subheader("Feature Comparison")
    if selected_features:
        st.line_chart(filtered_df[selected_features])
    
    # Histograms & Boxplots
    st.subheader("Feature Distributions")
    for feature in selected_features:
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        sns.histplot(filtered_df[feature], kde=True, ax=ax[0], color='skyblue')
        ax[0].set_title(f"{feature} Histogram")
        sns.boxplot(x=filtered_df[feature], ax=ax[1], color='lightgreen')
        ax[1].set_title(f"{feature} Boxplot")
        st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr_cols = selected_features + ["Outcome"] if "Outcome" not in selected_features else selected_features
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(filtered_df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# -----------------------------
# Model Training
elif section == "Model Training":
    st.subheader("Train & Compare Models")
    
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_choice = st.sidebar.multiselect("Select Models:", 
                                          ["Logistic Regression","Decision Tree","Random Forest","SVM"],
                                          default=["Logistic Regression","Random Forest"])
    
    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42)
    }
    
    results = {}
    for name in model_choice:
        model = models_dict[name]
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:,1]
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        results[name] = {"Accuracy": acc, "ROC-AUC": roc, "Model": model}
    
    st.write("### Model Performance Comparison")
    st.dataframe(pd.DataFrame(results).T[["Accuracy","ROC-AUC"]])
    
    if results:
        best_model_name = max(results, key=lambda x: results[x]["ROC-AUC"])
        best_model_local = results[best_model_name]["Model"]
        st.write(f"### Best Model: {best_model_name}")
        
        # Confusion Matrix
        y_pred = best_model_local.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
        # ROC Curve
        y_prob = best_model_local.predict_proba(X_test_scaled)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"{best_model_name} (AUC = {results[best_model_name]['ROC-AUC']:.3f})")
        ax.plot([0,1],[0,1],'--',color='gray')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

# -----------------------------
# Predict Diabetes for Custom Input
elif section == "Predict Diabetes":
    st.subheader("Predict Diabetes for Custom Input")
    
    st.write("Enter patient details:")
    input_data = {}
    for col in df.columns[:-1]:
        if col in ["Pregnancies","Age"]:
            input_data[col] = st.number_input(col, min_value=0, max_value=100, value=int(df[col].median()))
        else:
            input_data[col] = st.number_input(col, min_value=0.0, value=float(df[col].median()))
    
    if st.button("Predict"):
        X_input = pd.DataFrame([input_data])
        X_input_scaled = scaler.transform(X_input)
        y_pred_input = best_model.predict(X_input_scaled)[0]
        y_prob_input = best_model.predict_proba(X_input_scaled)[0][1]
        if y_pred_input == 1:
            st.error(f"Prediction: Diabetic (Probability: {y_prob_input:.2f})")
        else:
            st.success(f"Prediction: Non-Diabetic (Probability: {y_prob_input:.2f})")
