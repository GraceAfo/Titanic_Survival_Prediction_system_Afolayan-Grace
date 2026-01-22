import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("train.csv")  # Titanic dataset (Kaggle)

# Select 5 input features + target
features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
target = "Survived"

df = df[features + [target]]

# Handle missing values
# - Age: fill with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# - Embarked: fill with mode (most common)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Split data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
numeric_features = ["Pclass", "Age", "Fare"]
categorical_features = ["Sex", "Embarked"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Model
model = LogisticRegression(max_iter=1000)

# Pipeline (preprocessing + model)
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save the whole pipeline (so encoding/scaling is included)
joblib.dump(clf, "titanic_survival_model.pkl")
print("Model saved as titanic_survival_model.pkl")

# Demonstrate reload works
loaded_model = joblib.load("titanic_survival_model.pkl")
test_prediction = loaded_model.predict(X_test.iloc[:1])[0]
print("Reload test prediction (first test row):", test_prediction)
