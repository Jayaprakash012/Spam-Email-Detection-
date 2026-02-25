import streamlit as st
from joblib import load

# Load the best model and vectorizer
model_filename = 'SVM_linear_best_model.joblib'
vectorizer_filename = 'tfidf_vectorizer.joblib'

# Load the saved model and vectorizer
loaded_model = load(model_filename)
loaded_vectorizer = load(vectorizer_filename)

# Define a function for prediction
def predict_spam(input_text):
    # Transform the input text to feature vector
    input_features = loaded_vectorizer.transform([input_text])
    
    # Predict using the loaded model
    prediction = loaded_model.predict(input_features)[0]
    
    return 'Spam Mail' if prediction == 1 else 'Ham Mail'

# Create a Streamlit app
st.set_page_config(page_title="Spam Mail Detection", page_icon=":mailbox:", layout="wide")

# Custom CSS for enhanced styling with gradients
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    .main {
        background: linear-gradient(to right, #f7f9fc, #e0eafc);
        color: #333;
        font-family: 'Roboto', sans-serif;
        padding: 20px;
        height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .title {
        color: #004080;
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
        transition: transform 0.3s ease, color 0.3s ease;
    }
    .title:hover {
        transform: translateY(-10px);
        color: #002d72;
    }
    .stButton>button {
        background: linear-gradient(to right, #007bff, #0056b3);
        color: white;
        font-size: 18px;
        font-weight: 700;
        border-radius: 5px;
        padding: 12px 24px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #0056b3, #003d7a);
        transform: scale(1.05);
    }
    .stTextInput>div>div>input {
        font-size: 18px;
        border-radius: 5px;
        border: 2px solid #007bff;
        padding: 12px;
        transition: border-color 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #004080;
        outline: none;
    }
    .stMarkdown {
        font-size: 20px;
        font-weight: 500;
        color: #333;
    }
    .stSpinner {
        color: #007bff;
    }
    .result {
        font-size: 22px;
        font-weight: 700;
        color: #004080;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown("<div class='title'>Spam Mail Detection</div>", unsafe_allow_html=True)
st.write(
    "Welcome to the Spam Mail Detection app! Enter your email text below to check if it’s spam or not. "
    "The app uses a machine learning model to classify the email as 'Spam' or 'Ham'."
)

# Text input box for user to enter email
input_text = st.text_area("Email Text", height=300)

# Add a placeholder for displaying the result
result_placeholder = st.empty()

# When the user clicks the "Predict" button
if st.button('Predict'):
    if input_text:
        with st.spinner('Classifying...'):
            result = predict_spam(input_text)
            result_placeholder.markdown(f"<div class='result'>The input email is classified as: **{result}**</div>", unsafe_allow_html=True)
    else:
        result_placeholder.markdown("<div class='result'>Please enter some text in the text area.</div>", unsafe_allow_html=True)
