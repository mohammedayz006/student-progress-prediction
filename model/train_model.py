import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

data = pd.read_csv("../data/student_data.csv")

X = data[['study_hours', 'attendance', 'previous_marks', 'assignments']]
y = data['performance']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, "student_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model trained and saved successfully.")