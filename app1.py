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
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report

st.set_page_config(page_title="Diabetes Dashboard", layout="wide")
st.title(" Diabetes Prediction Dashboard")
st.write("Explore dataset, EDA, model training, and predictions")

# -----------------------------
# Load dataset
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
# Sidebar options
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", ["Dataset Preview", "EDA", "Model Training", "Predict Diabetes"])

# -----------------------------
# Dataset Preview
if section == "Dataset Preview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Class Distribution:")
    st.bar_chart(df["Outcome"].value_counts())

# -----------------------------
# Exploratory Data Analysis
elif section == "EDA":
    st.subheader("Exploratory Data Analysis")
    plot_option = st.sidebar.selectbox("Select Plot:", ["Class Distribution", "Feature Histograms", "Correlation Heatmap"])
    
    if plot_option == "Class Distribution":
        fig, ax = plt.subplots()
        sns.countplot(x="Outcome", data=df, ax=ax)
        ax.set_title("Class Distribution (0 = Non-Diabetic, 1 = Diabetic)")
        st.pyplot(fig)
    
    elif plot_option == "Feature Histograms":
        fig, ax = plt.subplots(figsize=(12,10))
        df.hist(ax=ax)
        st.pyplot(fig)
    
    elif plot_option == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

# -----------------------------
# Model Training & Evaluation
elif section == "Model Training":
    st.subheader("Model Training and Comparison")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Sidebar model selection
    model_choice = st.sidebar.multiselect("Select Model(s):", 
                                          ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"], 
                                          default=["Logistic Regression", "Random Forest"])
    
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
    
    st.write("### Model Comparison")
    st.dataframe(pd.DataFrame(results).T[["Accuracy","ROC-AUC"]])
    
    # Best model plots
    if results:
        best_model_name = max(results, key=lambda x: results[x]["ROC-AUC"])
        best_model = results[best_model_name]["Model"]
        st.write(f"### Best Model: {best_model_name}")
        
        # Confusion matrix
        y_pred = best_model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
        # ROC curve
        y_prob = best_model.predict_proba(X_test_scaled)[:,1]
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
# Predict diabetes for user input
elif section == "Predict Diabetes":
    st.subheader("Predict Diabetes for Custom Input")
    
    st.write("Enter patient details:")
    input_data = {}
    for col in df.columns[:-1]:
        if col in ["Pregnancies", "Age"]:
            input_data[col] = st.number_input(col, min_value=0, max_value=100, value=int(df[col].median()))
        else:
            input_data[col] = st.number_input(col, min_value=0.0, value=float(df[col].median()))
    
    if st.button("Predict"):
        X_input = pd.DataFrame([input_data])
        X_input_scaled = scaler.transform(X_input)
        # Use best model from training
        y_pred_input = best_model.predict(X_input_scaled)[0]
        y_prob_input = best_model.predict_proba(X_input_scaled)[0][1]
        if y_pred_input == 1:
            st.error(f"Prediction: Diabetic (Probability: {y_prob_input:.2f})")
        else:
            st.success(f"Prediction: Non-Diabetic (Probability: {y_prob_input:.2f})")
