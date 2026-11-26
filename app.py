# Import required libraries
import streamlit as st
import pandas as pd
import joblib


# Load the dataset
df=pd.read_csv('data/student.csv')
df=df.dropna()

input_cols=['Age', 'Gender', 'Family_Income', 'Internet_Access',
       'Study_Hours_per_Day', 'Attendance_Rate', 'Assignment_Delay_Days',
       'Travel_Time_Minutes', 'Part_Time_Job', 'Scholarship', 'Stress_Index',
       'GPA', 'Semester_GPA', 'CGPA', 'Semester', 'Department',
       'Parental_Education']


# load model
@st.cache_resource
def load_model():
    # load the model from a file
    model=joblib.load('models/dropout_models.pkl')
    return model

# Set page configuration

st.set_page_config(page_title="Student Dropout Prediction app",page_icon=":bar_chart:",layout="wide")

st.header("Student Dropout Prediction app")
st.text("This app predict the likelihood of a student dropping out based on various feature")

# Student input features
st.divider()
st.text_input("Enter Student Name")
age=st.slider("Enter Age",min_value=10,max_value=30,value=18)

col1,col2=st.columns(2)

with col1:
     gender=st.selectbox("Select gender",df.Gender.unique())
     internet=st.selectbox("Select the Internet",df.Internet_Access.unique())
     assignment_delays=st.selectbox("Assignment delay days",sorted(df.Assignment_Delay_Days.unique()))
     part_time=st.selectbox("Part Time Job",df.Part_Time_Job.unique())
     scholarship=st.selectbox("Select Scholarship",df.Scholarship.unique())
     semester=st.selectbox("Select Semester",df.Semester.unique())
     Department=st.selectbox("Select Department",df.Department.unique())
     parental_education=st.selectbox("Select Parental Education",df.Parental_Education.unique())



with col2:
     income=st.number_input("Select Family Income")
     study_hours=st.number_input("Enter the Study Hours Per Day",1.0,12.0,3.5)
     attendance_rate=st.slider("Enter attendance rate",1.0,100.0,50.0)
     travel_time = st.slider(
    "Enter Travel Time",
    float(df.Travel_Time_Minutes.min()),
    float(df.Travel_Time_Minutes.max()),
    float(df.Travel_Time_Minutes.mean()),)

     stress_index=st.slider("Enter Stress Index",0.0,10.0,df.Stress_Index.mean())
     Semester_GPA=st.slider("Select Semester GPA",df.Semester_GPA.min(),df.Semester_GPA.max(),df.Semester_GPA.mean())
     gpa=st.slider("Select GPA",df.GPA.min(),df.GPA.max(),df.GPA.mean())
     cgpa=st.slider("Select C GPA",df.CGPA.min(),df.CGPA.max(),df.CGPA.mean())


data=[[age,gender,internet,assignment_delays,part_time,scholarship,semester,Department,parental_education,income,study_hours,attendance_rate,travel_time,stress_index,Semester_GPA,gpa,cgpa]]

if st.button("Predict Dropout"):
     model=load_model()
     input_df=pd.DataFrame([{
          'Age': age,
        'Gender': gender,
        'Family_Income': income,
        'Internet_Access': internet,
        'Study_Hours_per_Day': study_hours,
        'Attendance_Rate': attendance_rate,
        'Assignment_Delay_Days': assignment_delays,
        'Travel_Time_Minutes': travel_time,
        'Part_Time_Job': part_time,
        'Scholarship': scholarship,
        'Stress_Index': stress_index,
        'GPA': gpa,
        'Semester_GPA': Semester_GPA,
        'CGPA': cgpa,
        'Semester': semester,
        'Department': Department,
        'Parental_Education': parental_education,
    }])[input_cols]
    
      
    
    
     y_pred=model.predict(input_df)
     if y_pred[0]==0:
          st.success("Not About to DropOut")
     else:
        st.error("He/She is about to dropout")
     

     
     st.dataframe(input_df)

     

