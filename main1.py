from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# CORS setup
origins = ["http://localhost", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

# Load models
POTATO_MODEL = tf.keras.models.load_model("models/cnn/1.h5")
CORN_MODEL = tf.keras.models.load_model("models/cnn/corn.keras")
APPLE_MODEL = tf.keras.models.load_model("models/cnn/apple.keras")


# Class names and remedies for potato
POTATO_CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
POTATO_INFO = {
    "Early Blight": {
        "causes": ["Caused by the fungus Alternaria solani.", "Favorable conditions: High humidity and temperatures between 24°C to 29°C (75°F to 85°F).", "Prolonged wetness from rain, dew, or irrigation.", "Weak plants due to poor nutrition, drought, or other stress factors.", "Infected plant debris in the soil.", "Spore dispersal through wind, rain, or irrigation."],
        "treatments": ["Use certified disease-free seed potatoes.", "Crop rotation with non-host crops.", "Apply fungicides like Mancozeb or Chlorothalonil.", "Remove and destroy infected plants.", "Ensure proper drainage and spacing.", "Apply Bordeaux mixture or copper-based fungicides."]
    },
    "Late Blight": {
        "causes": ["Caused by the oomycete pathogen Phytophthora infestans.", "Cool, wet weather with temperatures between 10°C to 20°C (50°F to 68°F).", "Prolonged leaf wetness from rain, dew, or irrigation.", "Spread through infected seed potatoes or contaminated soil.", "Spore dispersal by wind, water, or equipment."],
        "treatments": ["Use certified disease-free seed potatoes.", "Crop rotation with non-solanaceous crops.", "Apply fungicides such as Mancozeb or Metalaxyl.", "Remove and destroy infected plants.", "Ensure proper drainage and spacing.", "Introduce biological controls like Trichoderma harzianum."]
    },
    "Healthy": {
        "causes": ["No issues detected."],
        "treatments": ["Continue with regular care."]
    }
}

# Class names and remedies for corn
CORN_CLASS_NAMES = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy"
]
CORN_INFO = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "causes": ["Caused by the fungus Cercospora.", "Favorable conditions: High humidity and temperatures between 24°C to 29°C (75°F to 85°F).", "Prolonged wetness from rain, dew, or irrigation.", "Weak plants due to poor nutrition, drought, or other stress factors.", "Infected plant debris in the soil.", "Spore dispersal through wind, rain, or irrigation."],
        "treatments": ["Apply fungicides and improve drainage.", "Use resistant varieties and practice crop rotation."]
    },
    "Corn_(maize)___Common_rust_": {
        "causes": ["Caused by the fungus Puccinia sorghi.", "Favorable conditions: Warm temperatures and high humidity.", "Spore dispersal through wind and rain."],
        "treatments": ["Use resistant varieties.", "Apply fungicides and improve air circulation.", "Remove and destroy infected plant debris."]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "causes": ["Caused by the fungus Exserohilum turcicum.", "Favorable conditions: Cool, moist environments.", "Spore dispersal through wind and rain."],
        "treatments": ["Use resistant hybrids.", "Apply fungicides.", "Practice crop rotation and remove infected plants."]
    },
    "Corn_(maize)___healthy": {
        "causes": ["No issues detected."],
        "treatments": ["Continue with regular care."]
    }
}

# Class names and remedies for apple
APPLE_CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']
APPLE_INFO = {
    "Apple___Apple_scab": {
        "causes": ["Caused by the fungus Venturia inaequalis.", "Favorable conditions: Cool, wet weather.", "Spread through wind, rain, and irrigation."],
        "treatments": ["Apply fungicides like Captan or Mancozeb.", "Prune infected branches.", "Use resistant varieties.", "Ensure proper sanitation."]
    },
    "Apple___Black_rot": {
        "causes": ["Caused by the fungus Botryosphaeria obtusa.", "Favorable conditions: Warm, humid environments.", "Spread through wind, rain, and contaminated tools."],
        "treatments": ["Apply fungicides.", "Remove and destroy infected plant debris.", "Prune affected areas.", "Improve air circulation."]
    },
    "Apple___Cedar_apple_rust": {
        "causes": ["Caused by the fungus Gymnosporangium juniperi-virginianae.", "Spread by wind and rain from infected juniper trees.", "Favorable conditions: Cool, moist spring weather."],
        "treatments": ["Apply fungicides like Mancozeb.", "Remove nearby juniper trees if possible.", "Prune infected branches."]
    },
    "Apple___healthy": {
        "causes": ["No issues detected."],
        "treatments": ["Continue with regular care."]
    }
}

@app.get("/")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Query(..., enum=["potato", "corn", "apple"])
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    if model_type == "potato":
        model, class_names, info = POTATO_MODEL, POTATO_CLASS_NAMES, POTATO_INFO
    elif model_type == "corn":
        model, class_names, info = CORN_MODEL, CORN_CLASS_NAMES, CORN_INFO
    else:
        model, class_names, info = APPLE_MODEL, APPLE_CLASS_NAMES, APPLE_INFO

    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    causes = info[predicted_class]["causes"]
    treatments = info[predicted_class]["treatments"]

    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'causes': causes,
        'treatments': treatments
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
