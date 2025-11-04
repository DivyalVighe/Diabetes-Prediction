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





            st.success(f"Prediction: Non-Diabetic (Probability: {y_prob_input:.2f})")
