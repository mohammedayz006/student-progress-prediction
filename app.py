import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/student_model.pkl")
encoder = joblib.load("model/label_encoder.pkl")

st.title("Student Progress Prediction")

study_hours = st.number_input("Study hours per day", 0.0, 24.0)
attendance = st.number_input("Attendance percentage", 0.0, 100.0)
previous_marks = st.number_input("Previous exam marks", 0.0, 100.0)
assignments = st.number_input("Assignments score", 0.0, 100.0)

if st.button("Predict"):

    input_data = np.array(
        [[study_hours, attendance, previous_marks, assignments]]
    )

    prediction = model.predict(input_data)
    result = encoder.inverse_transform(prediction)

    st.success(f"Predicted Result : {result[0]}")