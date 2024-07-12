import streamlit as st
import pandas as pd
import os
import io
import requests
from dotenv import load_dotenv
import speech_recognition as sr
from streamlit_audio_recorder import audio_recorder

# Load environment variables from .env file
load_dotenv()

# Define API URL and API key
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B"
HF_API_TOKEN = "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"

# Function to load fine-tuning data and bot_score.csv
def load_data():
    fine_tuning_file_path = 'D:/bot/tune_data.txt'
    csv_file_path = 'D:/bot/bot_score.csv'

    fine_tuning_data = ""
    fitness_discount_data = {}

    if os.path.exists(fine_tuning_file_path):
        with open(fine_tuning_file_path, 'r') as file:
            fine_tuning_data = file.read()

    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        for index, row in df.iterrows():
            fitness_score = row['Fitness Score']
            discount = row['Discount']
            fitness_discount_data[fitness_score] = discount

    return fine_tuning_data, fitness_discount_data

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Function to predict discount based on fitness score
def predict_discount(fitness_score):
    if fitness_score >= 90:
        return 30  # 30% discount
    elif fitness_score >= 80:
        return 25  # 25% discount
    elif fitness_score >= 70:
        return 20  # 20% discount
    elif fitness_score >= 60:
        return 15  # 15% discount
    elif fitness_score >= 50:
        return 10  # 10% discount
    elif fitness_score >= 40:
        return 5   # 5% discount
    else:
        return 0   # No discount

# Function to generate AI assistant response
def generate_insurance_assistant_response(prompt_input, client, fine_tuning_data, fitness_discount_data):
    system_message = "You are a consultant with expertise in personal finance and insurance. Provide crisp and short responses."

    if fine_tuning_data:
        system_message += f"\n\nFine-tuning data:\n{fine_tuning_data}"

    if "fitness score" in prompt_input.lower() or "discount" in prompt_input.lower():
        return "Please provide your fitness score to get information about the discount you qualify for."

    try:
        user_fitness_score = float(prompt_input)
        discount = predict_discount(user_fitness_score)
        return f"Your fitness score is {user_fitness_score}. Based on this, you get {discount}% discount."
    except ValueError:
        pass

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt_input}
    ]

    response = ""
    for message in client.chat_completion(
            messages=messages,
            max_tokens=120,
            stream=True
    ):
        response += message.choices[0].delta.content or ""

    return response

def transcribe_speech():
    st.write("Click the microphone to start recording")
    audio_bytes = audio_recorder(text="", icon_size="2x")
    if audio_bytes:
        try:
            r = sr.Recognizer()
            audio_file = sr.AudioFile(io.BytesIO(audio_bytes))
            with audio_file as source:
                audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
            return None
    return None

# Define the AI Assistant page
def ai_assistant_page():
    st.title('AI Assistant')
    st.write("Your personal insurance and finance expert")

    # Custom CSS for chat containers
    st.markdown("""
    <style>
    .user-container {
        background-color: #2b5c8a;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .assistant-container {
        background-color: #1e3d5a;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .chat-text {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

    # Load data and configurations
    fine_tuning_data, fitness_discount_data = load_data()

    # Define sidebar for AI Assistant configurations
    with st.sidebar:
        st.title('🏛️🔍 AI-Assistant Settings')
        if HF_API_TOKEN:
            st.success('API key loaded from environment variable!', icon='✅')
        else:
            st.error('API key not found. Please set the HUGGINGFACE_API_TOKEN environment variable.', icon='🚨')

        # Initialize the InferenceClient
        client = InferenceClient(
            "meta-llama/Meta-Llama-3-8B",
            token=HF_API_TOKEN
        )

        st.button('Clear Chat History', on_click=clear_chat_history)

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            container_class = "user-container" if message['role'] == "user" else "assistant-container"
            st.markdown(f"""
            <div class="{container_class}">
                <p class="chat-text"><strong>{'You' if message['role'] == 'user' else 'Assistant'}:</strong> {message['content']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Speech-to-text option
    if st.button("Use Voice Input"):
        speech_text = transcribe_speech()
        if speech_text:
            st.session_state.messages.append({"role": "user", "content": speech_text})
            response = generate_insurance_assistant_response(speech_text, client, fine_tuning_data, fitness_discount_data)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.experimental_rerun()

    # Handle user input and generate responses
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Type your message here:", key="user_input")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = generate_insurance_assistant_response(user_input, client, fine_tuning_data, fitness_discount_data)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.experimental_rerun()

# Run the app
if __name__ == "__main__":
    ai_assistant_page()
