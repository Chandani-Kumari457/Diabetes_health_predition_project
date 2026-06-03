# ===============================
# train_models.py - Full Ready Code
# ===============================

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Step 2: Load Dataset
data = pd.read_csv(r"C:\Users\Chandani\Desktop\DMDW_PROJECT\diabetes.csv")
print("Dataset loaded ✅")
print(data.head())
print(data.info())
print(data.isnull().sum())

# Step 3: Feature Engineering (Advanced Touch)
# Age Group
data['AgeGroup'] = pd.cut(data['Age'], bins=[0,30,50,100], labels=['Young','Adult','Senior'])
# BMI Category
data['BMICategory'] = pd.cut(data['BMI'], bins=[0,18.5,25,30,100], labels=['Underweight','Normal','Overweight','Obese'])
print(data[['Age','AgeGroup','BMI','BMICategory']].head())

# Step 4: Features & Target
X = data.drop(["Outcome"], axis=1)  # Features
y = data["Outcome"]                 # Target
print("Features:", X.columns)
print("Target sample:", y.head())

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Step 6a: Logistic Regression (Baseline Model)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\n===== Logistic Regression =====")
print("Accuracy:", round(accuracy_score(y_test, lr_pred)*100,2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("Classification Report:\n", classification_report(y_test, lr_pred))

# Step 6b: Random Forest + Hyperparameter Tuning
params = {
    'n_estimators':[100,150],
    'max_depth':[None,5,10],
    'min_samples_split':[2,5],
    'min_samples_leaf':[1,2]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=params, cv=3)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
print("\n===== Random Forest =====")
print("Best Parameters:", grid.best_params_)

rf_pred = best_rf.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, rf_pred)*100,2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))

# Step 7: Feature Importance Plot (Optional Professional Touch)
importances = best_rf.feature_importances_
feat_names = X.columns

plt.figure(figsize=(10,6))
plt.barh(feat_names, importances)
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.show()

# Step 8: Save Best Random Forest Model
pickle.dump(best_rf, open(r"C:\Users\Chandani\Desktop\DMDW_PROJECT\diabetes_model.pkl", "wb"))
print("Best Random Forest model saved ✅")
