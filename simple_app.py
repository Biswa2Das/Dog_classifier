import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import json
import traceback

# Add debugging tools
DEBUG = True  # Set to False in production

# Configuration
st.set_page_config(
    page_title="Simple Dog Breed Identifier",
    page_icon="🐕",
    layout="wide"
)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "efficientnet_dog_classifier_final.pth"
BREED_INFO_PATH = "breed_info.json"
NUM_CLASSES = 120

# Basic CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        background-color: #F5F5F5;
    }
</style>
""", unsafe_allow_html=True)

# Cache helper functions
@st.cache_data
def load_class_names():
    """Load breed class names from directory or return placeholder data."""
    if os.path.exists("dog_pics"):
        return sorted(os.listdir("dog_pics"))
    else:
        # Return just 10 common breeds for demo
        return [
            "Labrador_retriever", "Golden_retriever", "Beagle", "Poodle", 
            "German_shepherd", "Bulldog", "Yorkshire_terrier", "Boxer", 
            "Dachshund", "Siberian_husky"
        ]

@st.cache_data
def load_breed_info():
    """Load breed information from JSON file or create placeholders."""
    if os.path.exists(BREED_INFO_PATH):
        with open(BREED_INFO_PATH, "r") as f:
            return json.load(f)
    else:
        # Create simple placeholder info
        breed_info = {}
        for breed in load_class_names():
            breed_info[breed] = {
                "description": f"The {breed.replace('_', ' ')} is a popular dog breed.",
                "temperament": "Varies",
                "life_span": "10-15 years",
            }
        return breed_info

@st.cache_resource
def load_model():
    """Load the PyTorch model or download it if not present."""
    try:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(MODEL_PATH) if os.path.dirname(MODEL_PATH) else '.', exist_ok=True)
        
        # Check if model exists locally
        if not os.path.exists(MODEL_PATH):
            st.warning("Model file not found. Running in demo mode with simulated predictions.")
            if DEBUG:
                st.info(f"Looking for model at: {os.path.abspath(MODEL_PATH)}")
            return None
            
        if DEBUG:
            st.info(f"Loading model from: {os.path.abspath(MODEL_PATH)}")
            
        model = models.efficientnet_b3(weights=None)
        num_ftrs = model.classifier[1].in_features
        
        # Using nn.Sequential with Dropout as shown in the original code
        # This matches the structure in your saved model
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, NUM_CLASSES)
        )
        
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        except Exception as e:
            if DEBUG:
                st.error(f"Error in state_dict: {str(e)}")
                st.code(traceback.format_exc())
            # Try with strict=False as fallback
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
            st.warning("Model loaded with strict=False, some layers may be missing")
            
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        if DEBUG:
            st.code(traceback.format_exc())
        return None

def get_image_transform():
    """Get the image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict_breed(image, model, class_names, topk=3):
    """Predict dog breed from image."""
    if model is None:
        # Demo mode - return simulated predictions
        import random
        selected_breeds = random.sample(class_names, min(topk, len(class_names)))
        probs = [random.random() for _ in range(topk)]
        total = sum(probs)
        probs = [p/total for p in probs]
        return list(zip(selected_breeds, sorted(probs, reverse=True)))
    
    # Process image
    image = image.convert("RGB")
    transform = get_image_transform()
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_idxs = torch.topk(probabilities, topk)
        top_probs = top_probs.cpu().numpy().flatten()
        top_idxs = top_idxs.cpu().numpy().flatten()
        top_classes = [class_names[idx] for idx in top_idxs]

    return list(zip(top_classes, top_probs))

# Main application
def main():
    st.markdown("<h1 class='main-header'>Simple Dog Breed Identifier 🐕</h1>", unsafe_allow_html=True)
    
    # Load resources
    class_names = load_class_names()
    breed_info = load_breed_info()
    model = load_model()
    
    # Display model status
    if model is None:
        st.warning("⚠️ Running in demo mode (model not found)")
    
    # Create sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Image", "About"])
    
    if page == "Upload Image":
        st.write("Upload a dog image to identify its breed")
        
        # Image upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Identify Breed"):
                with st.spinner("Analyzing..."):
                    predictions = predict_breed(image, model, class_names)
                
                # Display predictions
                st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                st.subheader("Prediction Results:")
                
                for i, (breed, probability) in enumerate(predictions):
                    st.write(f"{i+1}. {breed.replace('_', ' ')}")
                    st.progress(float(probability))
                    st.write(f"   Confidence: {probability * 100:.2f}%")
                    
                    # Show breed info in expander
                    with st.expander(f"About {breed.replace('_', ' ')}"):
                        if breed in breed_info:
                            info = breed_info[breed]
                            st.write(f"**Description:** {info['description']}")
                            st.write(f"**Temperament:** {info['temperament']}")
                            st.write(f"**Life Span:** {info['life_span']}")
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    else:  # About page
        st.subheader("About This App")
        st.write("""
        This simple app identifies dog breeds from uploaded images using an EfficientNet-B3 model.
        
        **Features:**
        - Upload and identify dog images
        - View breed information
        - See confidence scores for predictions
        
        **Technical Details:**
        - Model: EfficientNet-B3
        - Dataset: Stanford Dogs Dataset (120 breeds)
        - Built with Streamlit and PyTorch
        """)
        
        st.info("Images are processed locally and not stored permanently.")

# Run the app
if __name__ == "__main__":
    main()
