import tkinter as tk
from tkinter import messagebox
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load your dataset here
data = pd.read_csv('datasets/multi_data.csv')

# Function to display confusion matrix plots for different models
def plot_confusion_matrix(conf_matrix, model_name, classes):
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Model training and evaluation for Random Forest (Model 1)
def model_random_forest():
    X = data.drop(columns=['label'])
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_matrix, "Random Forest", np.unique(y))

# Model training and evaluation for Decision Tree (Model 2)
def model_decision_tree():
    X = data.drop(columns=['label'])
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_matrix, "Decision Tree", np.unique(y))

# Model training and evaluation for KNN (Model 3)
def model_knn():
    X = data.drop(columns=['label'])
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_matrix, "KNN", np.unique(y))

# Model training and evaluation for XGBoost (Model 4)
def model_xgboost():
    X = data.drop(columns=['label'])
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = XGBClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_matrix, "XGBoost", np.unique(y))

# Add a placeholder for model 6 (new model)
def model_six():
    # Assuming some classifier and training process for demonstration
    X = data.drop(columns=['label'])
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_matrix, "Model 6", np.unique(y))

# Function for displaying EDA plots
def display_eda():
    # Pie Chart: Distribution of normal and abnormal labels
    plt.figure(figsize=(8, 8))
    plt.pie(data['label'].value_counts(), labels=['normal', 'abnormal'], autopct='%0.2f%%')
    plt.title("Pie Chart: Normal vs Abnormal Labels", fontsize=16)
    plt.show()

    # Pie Chart: Distribution of multi-class attack categories
    plt.figure(figsize=(8, 8))
    plt.pie(data['attack_cat'].value_counts(), labels=data['attack_cat'].unique(), autopct='%0.2f%%')
    plt.title("Pie Chart: Multi-Class Attack Categories")
    plt.show()

# Function for displaying Model-related results
def display_model():
    model_random_forest()
    model_decision_tree()
    model_knn()
    model_xgboost()
    model_six()

# Main application window
root = tk.Tk()
root.title("Network Attack Monitoring System")
root.geometry("400x300")

# Create buttons for "Statistics" and "Model"
btn_statistics = tk.Button(root, text="Statistics", command=display_eda, width=20, height=2)
btn_statistics.pack(pady=20)

btn_model = tk.Button(root, text="Model", command=display_model, width=20, height=2)
btn_model.pack(pady=20)

# Run the application
root.mainloop()