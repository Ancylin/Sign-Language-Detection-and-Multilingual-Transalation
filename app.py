import json
import os
import streamlit as st
from twilio.rest import Client
from openai import OpenAI
import streamlit as st
import speech_recognition as sr
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
# import mediapipe as mp
import tensorflow as tf
import keras
import os
from deep_translator import GoogleTranslator
import sys
import json
from camera_input_live import camera_input_live
import os
import hashlib
import re
import numpy as np
import streamlit as st
from streamlit import session_state
from tensorflow.keras.models import load_model # type: ignore
from keras.preprocessing import image as img_preprocessing
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore
import base64
import tensorflow as tf
from PIL import Image
pose_sample_rpi_path = os.path.join(
    os.getcwd(), "examples/lite/examples/pose_estimation/raspberry_pi"
)
sys.path.append(pose_sample_rpi_path)
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import utils # type: ignore
from data import BodyPart # type: ignore
from playsound import playsound
from gtts import gTTS
from pathlib import Path
from ml import Movenet  # type: ignore
load_dotenv()
from time import sleep
session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0

st.set_page_config(
    page_title="Gesture Recognition",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)


class_names = np.array(
    [
        "all",
        "before",
        "black",
        "book",
        "candy",
        "chair",
        "clothes",
        "computer",
        "cousin",
        "deaf",
        "drink",
        "fine",
        "go",
        "help",
        "no",
        "thin",
        "walk",
        "who",
        "year",
        "yes",
    ]
)
signs = {
        "all": "All",
        "before": "Before",
        "black": "Black",
        "book": "Book",
        "candy": "Candy",
        "chair": "Chair",
        "clothes": "Clothes",
        "computer": "Computer",
        "cousin": "Cousin",
        "deaf": "Deaf",
        "drink": "Drink",
        "fine": "Fine",
        "go": "Go",
        "help": "Help",
        "no": "No",
        "thin": "Thin",
        "walk": "Walk",
        "who": "Who",
        "year": "Year",
        "yes": "Yes",
    }

@st.cache_resource
def load_models():
    model_dnn = tf.keras.models.load_model("models\\model.keras")
    movenet = Movenet("movenet_thunder")
    model_EfficientNetB0 = keras.models.load_model("models\\EfficientNetB0_model.keras")
    model_DenseNet169 = keras.models.load_model("models\\EfficientNetB0_model.keras")
    model_ResNet50 = keras.models.load_model("models\\EfficientNetB0_model.keras")
    model_InceptionV3 = keras.models.load_model("models\\EfficientNetB0_model.keras")
    return model_dnn, movenet, model_EfficientNetB0, model_DenseNet169, model_ResNet50, model_InceptionV3

model_dnn, movenet, model_EfficientNetB0, model_DenseNet169, model_ResNet50, model_InceptionV3 = load_models()
def play(text, language):
    try:
        text = translate(text,'en', language)
        speech = gTTS(text = text,lang=language, slow = False)
        speech.save('audio.mp3')
        audio_file = Path().cwd() /   'audio.mp3'
        playsound(audio_file)
        if os.path.exists('audio.mp3'):
            os.remove('audio.mp3')
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
def detect(input_tensor, inference_count=3):
    channel, image_height, image_width = input_tensor.shape
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor.numpy(), reset_crop_region=False)
    return person

def translate(text, source_language, target_language):
    texts = []
    while len(text) > 5000:
        index = 5000
        while text[index] != '.' and index > 0:
            index -= 1
        if index == 0:
            index = 5000
        texts.append(text[:index])
        text = text[index:]
    texts.append(text)
    translated_text = ""
    for text in texts:
        translated_text += GoogleTranslator(source=source_language, target=target_language).translate(text) + " "
    return translated_text

def processTensorImage(path):
    if type(path) == str:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image)
    else:
        image = tf.convert_to_tensor(path, dtype=tf.float32)
    return image


def classifySign(image):
    
    image = processTensorImage(image)
    person = detect(image)
    sign_landmarks = np.array(
        [
            [keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
            for keypoint in person.keypoints
        ],
        dtype=np.float32,
    )
    coordinates = sign_landmarks.flatten().astype(str).tolist()
    df = pd.DataFrame([coordinates]).reset_index(drop=True)
    X = df.astype("float64").to_numpy()
    y = model_dnn.predict(X)
    y_pred = [class_names[i] for i in np.argmax(y, axis=1)]
    return signs[y_pred[0]]

def predict(image_path, model):
    img = img_preprocessing.load_img(image_path, target_size=(224, 224))
    img_array = img_preprocessing.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    classes = ['All', 'Before', 'Black', 'Book', 'Candy', 'Chair', 'Clothes', 'Computer', 'Cousin', 'Deaf', 'Drink', 'Fine', 'Go', 'Help', 'No', 'Thin', 'Walk', 'Who', 'Year', 'Yes']
    return classes[np.argmax(predictions)]


def user_exists(email, json_file_path):
    # Function to check if user with the given email exists
    with open(json_file_path, "r") as file:
        users = json.load(file)
        for user in users["users"]:
            if user["email"] == email:
                return True
    return False


def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")
        if st.form_submit_button("Signup"):
            if not name:
                st.error("Name field cannot be empty.")
            elif not email:
                st.error("Email field cannot be empty.")
            elif not re.match(r"^[\w\.-]+@[\w\.-]+$", email):
                st.error("Invalid email format. Please enter a valid email address.")
            elif user_exists(email, json_file_path):
                st.error(
                    "User with this email already exists. Please choose a different email."
                )
            elif not age:
                st.error("Age field cannot be empty.")
            elif not password or len(password) < 6:  # Minimum password length of 6
                st.error("Password must be at least 6 characters long.")
            elif password != confirm_password:
                st.error("Passwords do not match. Please try again.")
            else:
                user = create_account(
                    name, email, age, sex, password, json_file_path
                )
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Signup successful. You are now logged in!")

def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)


        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                return user
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None

def initialize_database(
    json_file_path="data.json"
):
    try:
        if not os.path.exists(json_file_path):
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)

        
    except Exception as e:
        print(f"Error initializing database: {e}")

def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        email = email.lower()
        password = hashlib.md5(password.encode()).hexdigest()
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
        }

        data["users"].append(user_info)

        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None

def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Email:")
    password = st.text_input("Password:", type="password")
    password = hashlib.md5(password.encode()).hexdigest()
    username = username.lower()

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")

def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None

def render_dashboard(user_info, json_file_path="data.json"):
    try:
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")
        st.subheader("User Information:")
        st.write(f"Name: {user_info['name']}")
        st.write(f"Sex: {user_info['sex']}")
        st.write(f"Age: {user_info['age']}")
        
        st.image("https://www.stevenvanbelleghem.com/content/uploads/2023/11/19IcqVZ48A0tQba1-F_yIpg-820x540.jpeg")
        
    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")
    
def home_page():
    st.title("🖐️ Welcome to Sign Language Detection")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Breaking Communication Barriers with AI
        Our advanced sign language detection system uses cutting-edge machine learning 
        models to recognize and translate sign language in real-time. Whether through 
        images, videos, or webcam, we make communication more accessible for everyone.
        """)
        
        if not session_state.get("logged_in"):
            st.button("Get Started", key="get_started_btn", 
                     help="Create an account to start using our sign language detection")
    
    with col2:
        st.image("https://www.stevenvanbelleghem.com/content/uploads/2023/11/19IcqVZ48A0tQba1-F_yIpg-820x540.jpeg", 
                )
    
    # Features section
    st.markdown("---")
    st.header("🌟 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🎯 Multiple Input Methods
        - Image Upload
        - Video Processing
        - Real-time Webcam Detection
        """)
    
    with col2:
        st.markdown("""
        #### 🤖 Advanced AI Models
        - Hybrid DNN
        - EfficientNetB0
        - DenseNet169
        - ResNet50
        - InceptionV3
        """)
    
    with col3:
        st.markdown("""
        #### 🌍 Language Support
        - 100+ Languages
        - Text-to-Speech
        - Real-time Translation
        """)
    
    # How it works section
    st.markdown("---")
    st.header("🔄 How It Works")
    
    st.markdown("""
    1. **Choose Input Method**: Select from image, video, or webcam input
    2. **Select Model**: Pick from our range of advanced AI models
    3. **Choose Language**: Select your preferred output language
    4. **Get Results**: Receive real-time sign language detection and translation
    """)

def profile():
    if not session_state.get("logged_in"):
        st.warning("Please login to view your profile.")
        return
        
    st.title("👤 User Profile")
    user_info = session_state["user_info"]
    
    # Create a card-like container for profile info
    with st.container():
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://www.w3schools.com/howto/img_avatar.png", width=150)
            
        with col2:
            st.markdown(f"### {user_info['name']}")
            st.write(f"📧 Email: {user_info['email']}")
            st.write(f"🎂 Age: {user_info['age']}")
            st.write(f"⚧ Gender: {user_info['sex']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a logout button with a unique key
    if st.button("🚪 Logout", key="profile_logout_btn"):
        session_state["logged_in"] = False
        session_state["user_info"] = None
        st.success("Logged out successfully!")
        st.rerun()
def main():
    # Initialize session state for page access control
    if "logged_in" not in session_state:
        session_state["logged_in"] = False
        
    # Sidebar navigation
    with st.sidebar:
        st.title("🎯 Navigation")
        
        if session_state.get("logged_in"):
            page = st.radio(
                "Go to",
                ["Home", "Profile", "Sign Detection"],
                key="navigation"
            )
            
            if st.button("🚪 Logout"):
                session_state["logged_in"] = False
                session_state["user_info"] = None
                st.success("Logged out successfully!")
                st.rerun()
        else:
            page = "Login/Signup"
            st.info("Please login to access all features")
        
 
    
    # Main content based on navigation
    if page == "Home":
        home_page()
    elif page == "Login/Signup":
        login_or_signup = st.radio(
            "Select an option", 
            ("Login", "Signup"),
            horizontal=True,
            key="login_signup"
        )
        
        if login_or_signup == "Login":
            login()
        else:
            signup()
    elif page == "Profile":
        profile()
        
    elif page == "Sign Detection":
        if not session_state.get("logged_in"):
            st.warning("Please login to access Sign Detection.")
            return
            
        st.title("🖐️ Sign Language Detection")
        input_method = st.selectbox(
            "Select input method",
            ("---Select an input method---","Image", "Video", "Webcam"),
            key="input_method",
        )
        model = st.selectbox(
            "Select model",
            ("---Select a model---","Hybrid_DNN", "EfficientNetB0", "DenseNet169", "ResNet50", "InceptionV3"),
            key="model",
        )
        preferred_language = st.selectbox(
            "Select Language",
            (
                "English",
                "Hindi",
                "Afrikaans",
                "Albanian",
                "Amharic",
                "Arabic",
                "Armenian",
                "Azerbaijani",
                "Basque",
                "Belarusian",
                "Bengali",
                "Bosnian",
                "Bulgarian",
                "Catalan",
                "Cebuano",
                "Chichewa",
                "Chinese (simplified)",
                "Chinese (traditional)",
                "Corsican",
                "Croatian",
                "Czech",
                "Danish",
                "Dutch",
                "Esperanto",
                "Estonian",
                "Filipino",
                "Finnish",
                "French",
                "Frisian",
                "Galician",
                "Georgian",
                "German",
                "Greek",
                "Gujarati",
                "Haitian creole",
                "Hausa",
                "Hawaiian",
                "Hebrew",
                "Hmong",
                "Hungarian",
                "Icelandic",
                "Igbo",
                "Indonesian",
                "Irish",
                "Italian",
                "Japanese",
                "Javanese",
                "Kannada",
                "Kazakh",
                "Khmer",
                "Korean",
                "Kurdish (kurmanji)",
                "Kyrgyz",
                "Lao",
                "Latin",
                "Latvian",
                "Lithuanian",
                "Luxembourgish",
                "Macedonian",
                "Malagasy",
                "Malay",
                "Malayalam",
                "Maltese",
                "Maori",
                "Marathi",
                "Mongolian",
                "Myanmar (burmese)",
                "Nepali",
                "Norwegian",
                "Odia",
                "Pashto",
                "Persian",
                "Polish",
                "Portuguese",
                "Punjabi",
                "Romanian",
                "Russian",
                "Samoan",
                "Scots gaelic",
                "Serbian",
                "Sesotho",
                "Shona",
                "Sindhi",
                "Sinhala",
                "Slovak",
                "Slovenian",
                "Somali",
                "Spanish",
                "Sundanese",
                "Swahili",
                "Swedish",
                "Tajik",
                "Tamil",
                "Telugu",
                "Thai",
                "Turkish",
                "Ukrainian",
                "Urdu",
                "Uyghur",
                "Uzbek",
                "Vietnamese",
                "Welsh",
                "Xhosa",
                "Yiddish",
                "Yoruba",
                "Zulu",
            ),
        )
        languages = {
            "English": "en",
            "Afrikaans": "af",
            "Albanian": "sq",
            "Amharic": "am",
            "Arabic": "ar",
            "Armenian": "hy",
            "Azerbaijani": "az",
            "Basque": "eu",
            "Belarusian": "be",
            "Bengali": "bn",
            "Bosnian": "bs",
            "Bulgarian": "bg",
            "Catalan": "ca",
            "Cebuano": "ceb",
            "Chichewa": "ny",
            "Chinese (simplified)": "zh-cn",
            "Chinese (traditional)": "zh-tw",
            "Corsican": "co",
            "Croatian": "hr",
            "Czech": "cs",
            "Danish": "da",
            "Dutch": "nl",
            "Esperanto": "eo",
            "Estonian": "et",
            "Filipino": "tl",
            "Finnish": "fi",
            "French": "fr",
            "Frisian": "fy",
            "Galician": "gl",
            "Georgian": "ka",
            "German": "de",
            "Greek": "el",
            "Gujarati": "gu",
            "Haitian creole": "ht",
            "Hausa": "ha",
            "Hawaiian": "haw",
            "Hebrew": "he",
            "Hindi": "hi",
            "Hmong": "hmn",
            "Hungarian": "hu",
            "Icelandic": "is",
            "Igbo": "ig",
            "Indonesian": "id",
            "Irish": "ga",
            "Italian": "it",
            "Japanese": "ja",
            "Javanese": "jw",
            "Kannada": "kn",
            "Kazakh": "kk",
            "Khmer": "km",
            "Korean": "ko",
            "Kurdish (kurmanji)": "ku",
            "Kyrgyz": "ky",
            "Lao": "lo",
            "Latin": "la",
            "Latvian": "lv",
            "Lithuanian": "lt",
            "Luxembourgish": "lb",
            "Macedonian": "mk",
            "Malagasy": "mg",
            "Malay": "ms",
            "Malayalam": "ml",
            "Maltese": "mt",
            "Maori": "mi",
            "Marathi": "mr",
            "Mongolian": "mn",
            "Myanmar (burmese)": "my",
            "Nepali": "ne",
            "Norwegian": "no",
            "Odia": "or",
            "Pashto": "ps",
            "Persian": "fa",
            "Polish": "pl",
            "Portuguese": "pt",
            "Punjabi": "pa",
            "Romanian": "ro",
            "Russian": "ru",
            "Samoan": "sm",
            "Scots gaelic": "gd",
            "Serbian": "sr",
            "Sesotho": "st",
            "Shona": "sn",
            "Sindhi": "sd",
            "Sinhala": "si",
            "Slovak": "sk",
            "Slovenian": "sl",
            "Somali": "so",
            "Spanish": "es",
            "Sundanese": "su",
            "Swahili": "sw",
            "Swedish": "sv",
            "Tajik": "tg",
            "Tamil": "ta",
            "Telugu": "te",
            "Thai": "th",
            "Turkish": "tr",
            "Ukrainian": "uk",
            "Urdu": "ur",
            "Uyghur": "ug",
            "Uzbek": "uz",
            "Vietnamese": "vi",
            "Welsh": "cy",
            "Xhosa": "xh",
            "Yiddish": "yi",
            "Yoruba": "yo",
            "Zulu": "zu",
        }
        models = {
            "Hybrid_DNN": model_dnn,
            "EfficientNetB0": model_EfficientNetB0,
            "DenseNet169": model_DenseNet169,
            "ResNet50": model_ResNet50,
            "InceptionV3": model_InceptionV3,
        }
        if input_method != "---Select an input method---" and model != "---Select a model---":
            selected_model = models[model]
            if input_method == "Image":
                image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
                if image:
                    sign = ""
                    st.image(image)
                    if selected_model == model_dnn:
                        image = Image.open(image)
                        sign = classifySign(image)
                    else:
                        sign = predict(image, selected_model)
                    play(sign,languages[preferred_language])
                    st.success(f"Predicted Sign: {sign}")
            if input_method == "Video":
                video = st.file_uploader("Upload a video", type=["mp4"])
                if os.path.exists("output.mp4"):
                    st.write("TRUE")
                    os.remove("output.mp4")
                if video:
                    
                    st.video(video)
                    input_video = "input.mp4"
                    with open(input_video, "wb") as f:
                        f.write(video.getbuffer())
                    cap = cv2.VideoCapture(input_video)
                    frame_width = int(cap.get(3))
                    frame_height = int(cap.get(4))
                    out = cv2.VideoWriter(
                        "output.mp4",           
                        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                        10,
                        (frame_width, frame_height),
                    )
                    
                    if selected_model == model_dnn:
                        with st.spinner("Processing...."):
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                sign = classifySign(frame)
                                cv2.putText(frame, sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                                out.write(frame)
                    else:
                        with st.spinner("Processing...."):  
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                cv2.imwrite("temp.png", frame)
                                sign = predict("temp.png", selected_model)
                                if os.path.exists("temp.png"):
                                    os.remove("temp.png")
                                cv2.putText(frame, sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                                out.write(frame)
                            
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    os.system("ffmpeg -i output.mp4 -vcodec libx264 output_final.mp4 -y")
                    st.success("Video processing completed.")
                    
                    with open("output_final.mp4", "rb") as v:
                        st.video(v)
            if input_method == "Webcam":
                # if selected_model == model_dnn:
                image = camera_input_live()
                if image:
                    image.seek(0)
                    image = Image.open(image)
                    image = np.array(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    sign = classifySign(image)
                    image = cv2.flip(image, 1)
                    cv2.putText(
                        image,
                        sign,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
                    play(sign,languages[preferred_language])
                    st.image(image, channels="RGB")
                # else:
                #     image = camera_input_live()
                #     if image:
                #         image.seek(0)
                #         image = Image.open(image)
                #         image = np.array(image)
                #         image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                #         image = cv2.imread(image)
                #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #         sign = predict(image, selected_model)
                #         image = cv2.flip(image, 1)
                #         cv2.putText(
                #             image,
                #             sign,
                #             (50, 50),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             1,
                #             (255, 255, 255),
                #             2,
                #         )
                #         play(sign,languages[preferred_language])
                #         st.image(image, channels="RGB") 
        


if __name__ == "__main__":
    initialize_database()
    main()
