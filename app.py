import streamlit as st
import pandas as pd
from joblib import load
from streamlit_option_menu import option_menu

# Load the trained model
model_path = 'trained_model.joblib'
loaded_model = load(model_path)

# Function to set page background color
def set_background_color(color):
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the page configuration
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="📊",
    layout="wide"
)

# Set the background color for the entire app (default)
set_background_color("#f0f2f5")  # Light grey background

# Sidebar for navigation
with st.sidebar:
    st.title("Prediction App")
    page = option_menu(
        menu_title=None,
        options=["Home", "About", "Salary Prediction"],
        icons=["house", "info-circle", "graph-up-arrow"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": 'transparent'},
            "icon": {"color": "black", "font-size": "18px"},
            "nav-link": {"color": "black", "font-size": "16px", "text-align": "left", "margin": "0px", "margin-bottom": "15px"},
            "nav-link-selected": {"background-color": "#7B06A6", "font-size": "15px"},
        }
    )
    st.write("***")

if page == "Home":
    

    st.title("Welcome to the Salary Prediction App")
    st.write("This app predicts the minimum and maximum salary based on job title and minimum years of experience.")
    st.write("Navigate to the About page to learn more or to the Salary Prediction page to make predictions.")

elif page == "About":
    

    st.title("About This App")
    st.write("""\
        The Salary Prediction App is designed to provide minimum and maximum salary based on the following parameters:
        - Job Title: The title of the job position.
        - Minimum Years of Experience: The minimum years of relevant work experience.

        This app uses a machine learning model trained on historical salary data to make predictions.
    """)

elif page == "Salary Prediction":
    st.title('Salary Prediction')

    # Input fields for user
    job_title = st.selectbox('Job Title', [
        'Data Scientist', 'Business Analyst', 'Data Analyst',
        'Data Engineer', 'Senior Data Scientist',
        'Senior Business Analyst', 'Senior Data Analyst',
        'Senior Data Engineer', 'Machine Learning Engineer',
        'Data Architect'
    ])

    min_experience = st.number_input('Minimum Years of Experience', min_value=0, step=1)

    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'job_title': [job_title],
        'min_experience': [min_experience]
    })

    # One-hot encode the job_title
    input_data_encoded = pd.get_dummies(input_data, columns=['job_title'])

    # Ensure the input data has the same columns as the model's training data
    model_columns = loaded_model.feature_names_in_  # Feature names used in training
    input_data_encoded = input_data_encoded.reindex(columns=model_columns, fill_value=0)

    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: #7B06A6;  /* Purple background */
            color: white;  /* White text */
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
        }
        div.stButton > button:hover {
            background-color: #7B06A6 !important; /* Purple background on hover (no change) */
            color: white !important;
            border: none !important;
        }
        div.stButton > button:focus {
            background-color: #7B06A6 !important;  /* Maintain purple on focus */
            color: white !important;
            border: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    # Make prediction
    if st.button('Predict Salary'):
        # Make prediction
        prediction = loaded_model.predict(input_data_encoded)
        min_salary_in_inr = prediction[0][0]  # Minimum salary predicted
        max_salary_in_inr = prediction[0][1]  # Maximum salary predicted

        # Display the predicted salary range in two columns
        col1, col2 = st.columns(2)

        with col1:
            st.image("minimum_img.png", use_column_width=False, width=70)
            st.markdown(f"### Predicted Minimum Annual Salary")
            st.write(f'**₹{min_salary_in_inr:,.2f} INR**')

        with col2:
            st.image("maximum_image.png", use_column_width=False, width=70)  # Reduce image size by specifying width
            st.markdown(f"### Predicted Maximum Annual Salary")
            st.write(f'**₹{max_salary_in_inr:,.2f} INR**')
