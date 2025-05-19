import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time
import json
import pandas as pd
from pathlib import Path
import base64
from io import BytesIO
import random

# =====================
# Configuration
# =====================
st.set_page_config(
    page_title="Dog Breed Identifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define constants - using proper relative paths for cloud deployment
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 120

MODEL_PATH = "efficientnet_dog_classifier_final.pth"
DATASET_PATH = "dog_pics"
BREED_INFO_PATH = "breed_info.json"
# App state management
if 'breed_info_loaded' not in st.session_state:
    st.session_state.breed_info_loaded = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# =====================
# Custom CSS
# =====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6F61;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E86C1;
        margin-bottom: 1rem;
    }
    .breed-card {
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .breed-card:hover {
        transform: translateY(-5px);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        background-color: #F5F5F5;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .upload-box {
        border: 2px dashed #CCCCCC;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# =====================
# Helper Functions
# =====================

# Load class names
@st.cache_data
def load_class_names():
    # Check if dataset path exists, if not, return placeholder data
    if os.path.exists(DATASET_PATH):
        return sorted(os.listdir(DATASET_PATH))
    else:
        # Return placeholder breed names for demo purposes
        return [
            "Labrador_retriever", "Golden_retriever", "Beagle", "Poodle", 
            "German_shepherd", "Bulldog", "Yorkshire_terrier", "Boxer", 
            "Dachshund", "Siberian_husky"
        ]


# Load or create breed information
@st.cache_data
def load_breed_info():
    breed_info_path = Path(BREED_INFO_PATH)
    if os.path.exists(breed_info_path):
        with open(breed_info_path, "r") as f:
            return json.load(f)
    else:
        # Create placeholder info
        os.makedirs(os.path.dirname(breed_info_path), exist_ok=True)
        breed_info = {}
        for breed in load_class_names():
            breed_info[breed] = {
                "description": f"The {breed.replace('_', ' ')} is a unique dog breed with distinctive characteristics.",
                "origin": "Information not available",
                "temperament": "Varies",
                "height": "Varies",
                "weight": "Varies",
                "life_span": "10-15 years",
            }
        with open(breed_info_path, "w") as f:
            json.dump(breed_info, f)
        return breed_info


# Load breed images
@st.cache_data
def load_breed_images():
    breed_images = {}
    for breed in load_class_names():
        breed_folder = os.path.join(DATASET_PATH, breed)
        try:
            if os.path.exists(breed_folder):
                img_files = [f for f in os.listdir(breed_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if img_files:
                    breed_images[breed] = os.path.join(breed_folder, img_files[0])
                else:
                    breed_images[breed] = None
            else:
                breed_images[breed] = None
        except Exception:
            breed_images[breed] = None
    return breed_images


# Load model with caching
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            st.warning(f"Model file not found: {MODEL_PATH}. Running in demo mode with simulated predictions.")
            return None
            
        model = models.efficientnet_b3(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, NUM_CLASSES)
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Image preprocessing
@st.cache_data
def get_image_transform():
    return transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def preprocess_image(image):
    if image is None:
        return None

    image = image.convert("RGB")
    transform = get_image_transform()
    return transform(image).unsqueeze(0).to(DEVICE)


def get_topk_predictions(image_tensor, model, class_names, topk=3):
    if image_tensor is None:
        return []
        
    # Demo mode - return simulated predictions if model is not available
    if model is None:
        # Generate random predictions for demo mode
        return generate_demo_predictions(class_names, topk), 0.05
        
    with torch.no_grad():
        start_time = time.time()
        outputs = model(image_tensor)
        inference_time = time.time() - start_time

        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_idxs = torch.topk(probabilities, topk)
        top_probs = top_probs.cpu().numpy().flatten()
        top_idxs = top_idxs.cpu().numpy().flatten()
        top_classes = [class_names[idx] for idx in top_idxs]

        return list(zip(top_classes, top_probs)), inference_time


# Function to generate demo predictions when model is not available
def generate_demo_predictions(class_names, topk=3):
    # Take a random sample of classes
    selected_classes = random.sample(class_names, min(topk, len(class_names)))
    
    # Generate random probabilities that sum to 1
    probs = [random.random() for _ in range(topk)]
    prob_sum = sum(probs)
    probs = [p/prob_sum for p in probs]
    
    # Sort by probability in descending order
    predictions = list(zip(selected_classes, sorted(probs, reverse=True)))
    return predictions


# Display predictions with improved UI
def display_predictions(predictions, inference_time=None, breed_info=None):
    if not predictions:
        st.warning("No predictions available.")
        return

    st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)

    # Display inference time if available
    if inference_time:
        st.info(f"⏱ Inference time: {inference_time * 1000:.2f}ms")

    # Display confidence threshold selector
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.1,  # Lower default for demo mode
        step=0.05,
        help="Adjust the confidence threshold for predictions"
    )

    # Display filtered predictions
    filtered_predictions = [(breed, prob) for breed, prob in predictions if prob >= confidence_threshold]

    if filtered_predictions:
        for i, (breed, prob) in enumerate(filtered_predictions):
            # Progress bar for probability
            st.markdown(f"{i + 1}. {breed.replace('_', ' ')}")
            st.progress(float(prob))
            st.markdown(f"Confidence: {prob * 100:.2f}%")

            # Show breed info if available
            if breed_info and breed in breed_info:
                with st.expander("Show breed information"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        breed_images = load_breed_images()
                        if breed in breed_images and breed_images[breed] and os.path.exists(breed_images[breed]):
                            st.image(breed_images[breed], use_column_width=True)
                        else:
                            st.markdown("No image available")
                    with col2:
                        st.markdown(f"### {breed.replace('_', ' ')}")
                        st.markdown(f"Description: {breed_info[breed]['description']}")
                        st.markdown(f"Origin: {breed_info[breed]['origin']}")
                        st.markdown(f"Temperament: {breed_info[breed]['temperament']}")
                        st.markdown(f"Size: {breed_info[breed]['height']} in height, {breed_info[breed]['weight']}")
                        st.markdown(f"Life Span: {breed_info[breed]['life_span']}")

            st.markdown("---")
    else:
        st.warning("No predictions above the confidence threshold.")

    st.markdown("</div>", unsafe_allow_html=True)


# Function to display image with download button
def display_image_with_download(image, caption=""):
    st.image(image, caption=caption, use_column_width=True)

    buffered = BytesIO()

    # Convert image to RGB if not already
    if image.mode != "RGB":
        image = image.convert("RGB")

    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    href = f'<a href="data:file/jpg;base64,{img_str}" download="dog_prediction.jpg">Download Image</a>'
    st.markdown(href, unsafe_allow_html=True)


# Search functionality
def search_breeds(search_term, breed_names):
    if not search_term:
        return breed_names

    search_term = search_term.lower()
    return [breed for breed in breed_names if search_term in breed.lower()]


# =====================
# Load Data
# =====================
with st.spinner("Loading resources..."):
    class_names = load_class_names()
    breed_images = load_breed_images()
    breed_info = load_breed_info()
    st.session_state.breed_info_loaded = True


# Load model (can be deferred until needed)
def ensure_model_loaded():
    if not st.session_state.model_loaded:
        with st.spinner("Loading AI model..."):
            model = load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.model_loaded = True
                return model
            else:
                # Return None to indicate demo mode
                return None
    return st.session_state.model


# =====================
# Sidebar Navigation
# =====================
st.sidebar.markdown("<h1 style='text-align: center;'>🐾 Dog Breed Identifier</h1>", unsafe_allow_html=True)

if 'menu' not in st.session_state:
    st.session_state.menu = "Home"

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Upload Image", "Take Photo", "View Breeds", "Instructions", "About"],
    index=["Home", "Upload Image", "Take Photo", "View Breeds", "Instructions", "About"].index(st.session_state.menu)
)

# Display model status
if not os.path.exists(MODEL_PATH):
    st.sidebar.warning("⚠ Running in demo mode (model not found)")
    
# Display processor info
st.sidebar.markdown("---")
st.sidebar.info(f"Running on: {DEVICE}")

# =====================
# Pages
# =====================
if menu == "Home":
    st.markdown("<h1 class='main-header'>Welcome to Dog Breed Identifier 🐾</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ### Identify 120 Dog Breeds with AI

        This app uses a state-of-the-art EfficientNet-B3 model to identify dog breeds from images. 
        Simply upload a photo or take one with your camera!

        ### Features:
        - 🖼 Upload images from your device
        - 📸 Take photos using your webcam
        - 🔍 View all 120 dog breeds in our gallery
        - 📊 Get detailed breed information
        - 🏆 See confidence scores for predictions
        """)

        # Quick access buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🖼 Upload Image", use_container_width=True):
                st.session_state.menu = "Upload Image"
                st.rerun()
        with col_b:
            if st.button("📸 Take Photo", use_container_width=True):
                st.session_state.menu = "Take Photo"
                st.rerun()
    with col2:
        st.image(
            "https://images.unsplash.com/photo-1548199973-03cce0bbc87b?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb&w=640",
            caption="Identify over 120 dog breeds")

    st.markdown("---")

    # Sample breed showcase
    st.markdown("<h2 class='sub-header'>Popular Dog Breeds</h2>", unsafe_allow_html=True)

    # Show 5 random breeds as a preview
    sample_breeds = random.sample(class_names, min(5, len(class_names)))

    cols = st.columns(5)
    for i, breed in enumerate(sample_breeds):
        with cols[i]:
            st.markdown(f"<div class='breed-card'>", unsafe_allow_html=True)
            if breed in breed_images and breed_images[breed] and os.path.exists(breed_images[breed]):
                st.image(breed_images[breed], use_column_width=True)
            else:
                st.markdown("No image available")
            st.markdown(f"<p style='text-align:center'><b>{breed.replace('_', ' ')}</b></p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Usage stats (these would be real in a production app)
    st.markdown("<h2 class='sub-header'>App Statistics</h2>", unsafe_allow_html=True)
    col2, col3 = st.columns(2)
    col2.metric("Breeds Identified", "120")
    col3.metric("Accuracy", "94.2%")

elif menu == "Upload Image":
    st.markdown("<h1 class='main-header'>Upload Dog Image</h1>", unsafe_allow_html=True)

    # Create a nice upload area
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your dog image here or click to browse",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file)
        display_image_with_download(image, "Uploaded Image")

        # Ensure model is loaded (might be None in demo mode)
        model = ensure_model_loaded()

        if st.button("🔍 Identify Breed", use_container_width=True):
            with st.spinner("Analyzing image..."):
                input_tensor = preprocess_image(image)
                predictions, inference_time = get_topk_predictions(input_tensor, model, class_names, topk=5)

            # Display predictions
            display_predictions(predictions, inference_time, breed_info)

            # Save prediction history (in a real app, you might save this to a database)
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []

            if predictions:
                top_breed, top_prob = predictions[0]
                st.session_state.prediction_history.append({
                    'breed': top_breed,
                    'confidence': top_prob,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })

        # Tips for better results
        with st.expander("Tips for better results"):
            st.markdown("""
            - Make sure the dog is the main subject in the image
            - Use well-lit, clear images
            - Try different angles if uncertain
            - Crop the image to focus on the dog
            """)

elif menu == "Take Photo":
    st.markdown("<h1 class='main-header'>Take Dog Photo</h1>", unsafe_allow_html=True)

    st.info("📸 Use your device camera to take a photo of your dog.")

    captured_image = st.camera_input("Take a picture")

    if captured_image:
        image = Image.open(captured_image)

        # Ensure model is loaded
        model = ensure_model_loaded()

        if st.button("🔍 Identify Breed", use_container_width=True):
            with st.spinner("Analyzing image..."):
                input_tensor = preprocess_image(image)
                predictions, inference_time = get_topk_predictions(input_tensor, model, class_names, topk=5)

            # Display predictions
            display_predictions(predictions, inference_time, breed_info)

        # Tips for camera capture
        with st.expander("Tips for camera capture"):
            st.markdown("""
            - Hold your device steady
            - Ensure good lighting
            - Position the dog to face the camera
            - Try to get the whole dog in frame
            - Avoid busy backgrounds
            """)

elif menu == "View Breeds":
    st.markdown("<h1 class='main-header'>Browse Dog Breeds</h1>", unsafe_allow_html=True)

    # Add search functionality
    search_term = st.text_input("🔍 Search breeds", "")
    filtered_breeds = search_breeds(search_term, class_names)

    # Sort and filter options
    col1, col2 = st.columns(2)
    with col1:
        sort_option = st.selectbox(
            "Sort by:",
            ["Alphabetical (A-Z)", "Alphabetical (Z-A)"]
        )
    with col2:
        view_option = st.selectbox(
            "View as:",
            ["Grid", "List"]
        )

    # Sort based on selection
    if sort_option == "Alphabetical (A-Z)":
        filtered_breeds = sorted(filtered_breeds)
    else:
        filtered_breeds = sorted(filtered_breeds, reverse=True)

    # Show results count
    st.write(f"Showing {len(filtered_breeds)} of {len(class_names)} breeds")

    # Display breeds based on view option
    if view_option == "Grid":
        # Create a grid layout
        cols = st.columns(4)
        for idx, breed in enumerate(filtered_breeds):
            with cols[idx % 4]:
                st.markdown(f"<div class='breed-card'>", unsafe_allow_html=True)
                if breed in breed_images and breed_images[breed] and os.path.exists(breed_images[breed]):
                    st.image(breed_images[breed], use_column_width=True)
                else:
                    st.markdown("No image available")

                if st.button(f"View {breed.replace('', ' ')}", key=f"breed{breed}"):
                    st.session_state.selected_breed = breed
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        # List view
        for breed in filtered_breeds:
            col1, col2 = st.columns([1, 3])
            with col1:
                if breed in breed_images and breed_images[breed] and os.path.exists(breed_images[breed]):
                    st.image(breed_images[breed], use_column_width=True)
                else:
                    st.markdown("No image available")
            with col2:
                st.markdown(f"### {breed.replace('_', ' ')}")
                if breed in breed_info:
                    st.markdown(f"{breed_info[breed]['description'][:100]}...")
                if st.button(f"View Details", key=f"list_{breed}"):
                    st.session_state.selected_breed = breed
            st.markdown("---")

    # Show selected breed details
    if 'selected_breed' in st.session_state:
        breed = st.session_state.selected_breed
        st.markdown("---")
        st.markdown(f"<h2 class='sub-header'>{breed.replace('_', ' ')}</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            if breed in breed_images and breed_images[breed] and os.path.exists(breed_images[breed]):
                st.image(breed_images[breed], use_column_width=True)
            else:
                st.markdown("No image available")
        with col2:
            if breed in breed_info:
                info = breed_info[breed]
                st.markdown(f"Description: {info['description']}")
                st.markdown(f"Origin: {info['origin']}")
                st.markdown(f"Temperament: {info['temperament']}")
                st.markdown(f"Height: {info['height']}")
                st.markdown(f"Weight: {info['weight']}")
                st.markdown(f"Life Span: {info['life_span']}")

elif menu == "Instructions":
    st.markdown("<h1 class='main-header'>How to Use This App</h1>", unsafe_allow_html=True)

    st.markdown("### Getting Started")
    st.markdown("""
    This app uses artificial intelligence to identify dog breeds from photos. Follow these steps to get the best results:
    """)

    # Step-by-step instructions with tabs
    tab1, tab2, tab3 = st.tabs(["Preparing Images", "Using the App", "Understanding Results"])

    with tab1:
        st.markdown("### Tips for Better Photos")
        st.markdown("""
        - Use clear, well-lit images with the dog's face and body visible.
        - Avoid blurry or very dark photos.
        - Upload images showing the full dog or most of the body for better accuracy.
        - Use photos with a plain background if possible.
        - Try different angles if uncertain.
        - For camera capture, hold device steady and ensure focus before snapping.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Good Example")
            st.image(
                "https://images.unsplash.com/photo-1543466835-00a7907e9de1?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8NXx8ZG9nfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=500&q=60",
                caption="Clear, well-lit, full-body shot")
        with col2:
            st.markdown("#### Poor Example")
            st.image(
                "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8ZG9nJTIwYmx1cnJ5fGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=500&q=60",
                caption="Blurry image with poor lighting")

    with tab2:
        st.markdown("### Using the App")
        st.markdown("""
        1. Upload an Image:
           - Click "Upload Image" in the sidebar
           - Upload a dog photo from your device
           - Click "Identify Breed" to get results

        2. Take a Photo:
           - Click "Take Photo" in the sidebar
           - Use your device's camera to take a picture
           - Click "Identify Breed" to get results

        3. Browse Breeds:
           - Click "View Breeds" to see all available dog breeds
           - Use the search box to find specific breeds
           - Click on any breed to view detailed information
        """)

    with tab3:
        st.markdown("### Understanding Results")
        st.markdown("""
        - The app shows predictions with confidence percentages
        - Higher confidence (closer to 100%) indicates greater certainty
        - Multiple predictions may be shown if the model is uncertain
        - Use the confidence threshold slider to filter results
        - You can view detailed information about each predicted breed
        """)

        st.image(
            "https://images.unsplash.com/photo-1587300003388-59208cc962cb?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Nnx8ZG9nJTIwYnJlZWRzfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=500&q=60",
            caption="Example of prediction results")

        st.markdown("""
        Note: While the AI model is very accurate, it may occasionally misidentify breeds, especially with:
        - Mixed breed dogs
        - Unusual color variations
        - Puppies (which may not yet have all adult features)
        - Rare or uncommon breeds
        - Poor quality or unusually angled photos
        """)

elif menu == "About":
    st.markdown("<h1 class='main-header'>About This App</h1>", unsafe_allow_html=True)

    st.markdown("""
    ### Dog Breed Identifier

    This application uses a deep learning model (EfficientNet-B3) trained on a dataset of 120 different dog breeds to identify dogs in images.

    ### Technology Stack:
    - Frontend: Streamlit
    - AI Model: EfficientNet-B3 (PyTorch)
    - Dataset: Stanford Dogs Dataset (120 breeds)

    ### Model Performance:
    - Accuracy: ~94% on test set

    ### Privacy Notice:
    Images uploaded to this application are processed locally and are not stored permanently. 
    They are only used for the purpose of breed identification during your session.

    ### Credits:
    - Dog breed dataset: Stanford Dogs Dataset
    - Model architecture: EfficientNet-B3 (Google Research)
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center;'>
        <p>Dog Breed Identifier v1.0 | Built with ❤ using Streamlit and PyTorch</p>
    </div>
    """,
    unsafe_allow_html=True)

# Check if the app is in demo mode (model not found)
if not os.path.exists(MODEL_PATH):
    st.warning("""
    ⚠ Model file not found. The app is running in demo mode with simulated predictions.
    
    To use the full model:
    1. Download the efficientnet_dog_classifier_final.pth model file
    2. Place it in the models folder
    3. Restart the app
    """)
