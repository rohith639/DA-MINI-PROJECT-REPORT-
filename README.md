# DA-MINI-PROJECT-REPORT-
program :
# ================================
# Data Collection and Preprocessing (preprocessing.py)
# ================================
    import pandas as pd
    import numpy as np
    def load_student_data(file_path):
     # Load dataset
     df = pd.read_csv(file_path)
     # Remove missing values
     df = df.dropna()
     # Remove duplicates
     df = df.drop_duplicates()
     return df
    def preprocess_data(df):
     df = df.copy()
     # Convert categorical columns into numeric
     df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
     df['Scholarship'] = df['Scholarship'].map({'Yes': 1, 'No': 0})
     # Create target column
     df['Dropout'] = df['Dropout'].map({'Yes': 1, 'No': 0})
     # Risk Score Feature
     df['Risk_Score'] = (
        (100 - df['Attendance']) * 0.4 +
        (10 - df['GPA']) * 5 +
        df['Backlogs'] * 2
     )

    return df
# ================================
# Model Training (model.py)
# ================================

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    def train_model(df):
      features = [
        'Age',
        'Gender',
        'Attendance',
        'GPA',
        'Income',
        'Scholarship',
        'Backlogs',
        'Risk_Score'
     ]
     X = df[features]
     y = df['Dropout']
     X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
     )
     model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
     )
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     acc = accuracy_score(y_test, y_pred)
     cm = confusion_matrix(y_test, y_pred)
     print("Accuracy:", round(acc, 2))
     print("Confusion Matrix:\n", cm)
     print(classification_report(y_test, y_pred))
    return model
    # ================================
# Visualization (visualization.py)
# ================================

    import matplotlib.pyplot as plt
    import seaborn as sns
    def plot_attendance_dropout(df):
     plt.figure(figsize=(8,5))
     sns.boxplot(x='Dropout', y='Attendance', data=df)
     plt.title("Attendance vs Dropout")
     plt.xlabel("Dropout (0=No,1=Yes)")
     plt.ylabel("Attendance %")
     plt.show()
    def plot_gpa_dropout(df):
     plt.figure(figsize=(8,5))
     sns.boxplot(x='Dropout', y='GPA', data=df)
     plt.title("GPA vs Dropout")
     plt.xlabel("Dropout")
     plt.ylabel("GPA")
     plt.show()
    def plot_correlation_heatmap(df):
     plt.figure(figsize=(10,8))
     sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
     plt.title("Feature Correlation Heatmap")
     plt.show()
    def plot_dropout_count(df):
     plt.figure(figsize=(6,4))
     sns.countplot(x='Dropout', data=df)
     plt.title("Dropout Count")
     plt.xlabel("Dropout")
     plt.ylabel("Count")
     plt.show()
     # ================================
# Flask Backend (app.py)
# ================================
    from flask import Flask, render_template, request
    import pickle
    import pandas as pd

    app = Flask(__name__)
    model = pickle.load(open("model.pkl", "rb"))
    @app.route('/')
    def home():
       return render_template("index.html")
     @app.route('/predict', methods=['POST'])
    def predict():
      data = [
        int(request.form['age']),
        int(request.form['gender']),
        float(request.form['attendance']),
        float(request.form['gpa']),
        float(request.form['income']),
        int(request.form['scholarship']),
        int(request.form['backlogs']),
        float(request.form['risk_score'])
     ]
     prediction = model.predict([data])[0]
     result = "Likely to Dropout" if prediction == 1 else "Continue Studies"
     return render_template("index.html", result=result)
    if __name__ == "__main__":
       app.run(debug=True)
       # ================================
# Main Execution (main.py)
# ================================
    from preprocessing import load_student_data, preprocess_data
    from model import train_model
    from visualization import (
    plot_attendance_dropout,
    plot_gpa_dropout,
    plot_correlation_heatmap,
    plot_dropout_count
     )
    import pickle
    # Step 1: Load dataset
    df = load_student_data("students.csv")
    # Step 2: Preprocess data
    df = preprocess_data(df)
    # Step 3: Train model
    model = train_model(df)
    # Step 4: Save model
    pickle.dump(model, open("model.pkl", "wb"))
    # Step 5: Generate graphs
    plot_attendance_dropout(df)
    plot_gpa_dropout(df)
    plot_correlation_heatmap(df)
    plot_dropout_count(df)
