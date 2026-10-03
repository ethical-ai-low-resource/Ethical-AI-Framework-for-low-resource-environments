import streamlit as st
import joblib
import os

# Page configuration for clean and responsive UI
st.set_page_config(
    page_title="Urdu Text Classifier",
    page_icon="✍️",
    layout="centered"
)

# Custom CSS for clean layout and right-to-left Urdu text support
st.markdown("""
    <style>
    .stTextArea textarea {
        direction: rtl;
        text-align: right;
        font-family: 'Naheed', 'Noto Naskh Arabic', sans-serif;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)


# Function to safely load pre-trained artifacts using Streamlit caching
@st.cache_resource
def load_pipeline_artifacts():
    # Checking if required model files exist in the project directory
    model_path = 'urdu_model.pkl'
    vectorizer_path = 'vectorizer.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None, f"Artifacts missing: Make sure '{model_path}' and '{vectorizer_path}' exist in the directory."
    
    try:
        # Loading saved trained model and vectorizer
        loaded_model = joblib.load(model_path)
        loaded_vectorizer = joblib.load(vectorizer_path)
        return loaded_model, loaded_vectorizer, None
    except Exception as err:
        return None, None, f"Failed to load artifacts: {str(err)}"

# Main App Title and Header
st.title("✍️ Urdu Text Classification System")
st.caption("A lightweight baseline ML pipeline powered by TF-IDF & Logistic Regression.")

# Load models and handle missing file errors gracefully
model, vectorizer, error_message = load_pipeline_artifacts()

if error_message:
    # Stop execution if model files are missing or unreadable
    st.error(error_message)
    st.stop()

# Input UI Box
st.subheader("Urdu Input Text")
user_text = st.text_area(
    label="Enter text for classification:",
    placeholder="yahan urdu text likhein...",
    height=140
)

# Predict Button and Processing Logic
if st.button("Classify Text", type="primary", use_container_width=True):
    # Validating input to prevent empty string processing
    cleaned_input = user_text.strip()
    
    if not cleaned_input:
        st.warning("Pehlay kuch text enter karein phir classify karein.")
    else:
        try:
            with st.spinner("Processing text and predicting..."):
                # Step 1: Vectorize input text using pre-fitted TF-IDF
                text_vector = vectorizer.transform([cleaned_input])
                
                # Step 2: Model prediction and probability scores
                prediction = model.predict(text_vector)[0]
                prediction_probs = model.predict_proba(text_vector)[0]
                
                # Step 3: Extract maximum confidence percentage
                confidence_score = max(prediction_probs) * 100

            # Displaying Results in Clean UI Layout
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="Predicted Category", value=str(prediction))
            
            with col2:
                st.metric(label="Model Confidence", value=f"{confidence_score:.1f}%")

        except Exception as ex:
            st.error(f"Prediction ke doran error aya: {str(ex)}")

# Subtle footer
st.markdown("---")
st.caption("Developed for Low-Resource Urdu NLP Research & Benchmarking.")