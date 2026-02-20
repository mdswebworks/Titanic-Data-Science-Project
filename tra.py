import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("tested.csv.csv") 

# Drop Cabin
df.drop("Cabin", axis=1, inplace=True, errors='ignore')

# Fill missing values for numeric columns
df['Age'] = df['Age'].fillna(df['Age'].median()) # Median is often safer than mean
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Handle Embarked missing values BEFORE mapping
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convert categorical to numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Select features
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = df['Survived']

# Final Check: If any NaNs remain, drop those rows (emergency fallback)
X = X.fillna(0) 

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and Evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))