"""Implimenting Random Forest AI Model For Titanic Incident : Predicting if a person wil
live or not"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# -------------------------
# Load the Titanic dataset
# -------------------------
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(URL)

# -------------------------
# SAVE the dataset locally
# -------------------------
df.to_csv("titanic_raw.csv", index=False)
print("Raw Titanic dataset saved as titanic_raw.csv\n")

print("Initial preview of data:")
print(df.head())

# -------------------------
# Preprocessing
# -------------------------

# Remove columns not useful for prediction
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Fix missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encode categorical variables
cat_cols = ['Sex', 'Embarked']
encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# -------------------------
# Split data
# -------------------------
X = df.drop('Survived', axis=1)
y = df['Survived']

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    X, y, test_size=0.2, random_state=42)

# -------------------------
# Train Random Forest
# -------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_TRAIN, Y_TRAIN)

# -------------------------
# Evaluate
# -------------------------
pred = model.predict(X_TEST)

print("\nAccuracy:", accuracy_score(Y_TEST, pred))
print("\nClassification Report:\n", classification_report(Y_TEST, pred))

# -------------------------
# Feature Importance
# -------------------------
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importance:\n", importances.sort_values(ascending=False))
