import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# -------------------------
# Load the Titanic dataset
# -------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# -------------------------
# Train Random Forest
# -------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# -------------------------
# Feature Importance
# -------------------------
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importance:\n", importances.sort_values(ascending=False))
