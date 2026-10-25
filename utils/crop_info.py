"""
crop_info.py
------------
Static reference information about the 22 crops in the dataset:
ideal growing conditions, category, and a short description.

This is used by the Streamlit app for the "Crop Information" section
and for the "Crop Comparison" feature. It is hand-curated, general
agronomic knowledge and should be validated against local extension
services before being used for real planting decisions.
"""

CROP_INFO = {
    "rice": {
        "category": "Cereal",
        "description": "A staple cereal crop that thrives in warm, humid climates with standing water or heavy rainfall.",
        "ideal_conditions": {"N": "80-100", "P": "40-60", "K": "35-45", "temperature": "20-27°C", "humidity": "75-90%", "ph": "5.5-7.0", "rainfall": "180-300mm"},
    },
    "maize": {
        "category": "Cereal",
        "description": "A versatile cereal grown widely for food, feed, and industrial use; prefers well-drained soil and moderate rainfall.",
        "ideal_conditions": {"N": "60-100", "P": "35-60", "K": "15-25", "temperature": "18-27°C", "humidity": "55-75%", "ph": "5.5-7.0", "rainfall": "60-110mm"},
    },
    "chickpea": {
        "category": "Legume",
        "description": "A cool-season legume (gram) that fixes nitrogen in the soil and tolerates dry conditions well.",
        "ideal_conditions": {"N": "20-60", "P": "60-80", "K": "75-85", "temperature": "17-21°C", "humidity": "14-20%", "ph": "6.0-7.5", "rainfall": "65-95mm"},
    },
    "kidneybeans": {
        "category": "Legume",
        "description": "A protein-rich legume that prefers cooler temperatures and moderate rainfall, sensitive to waterlogging.",
        "ideal_conditions": {"N": "20-40", "P": "60-80", "K": "15-25", "temperature": "15-25°C", "humidity": "18-24%", "ph": "5.5-6.5", "rainfall": "60-150mm"},
    },
    "pigeonpeas": {
        "category": "Legume",
        "description": "A hardy, drought-tolerant legume commonly intercropped, valued for its nitrogen-fixing ability.",
        "ideal_conditions": {"N": "0-40", "P": "60-80", "K": "15-25", "temperature": "18-37°C", "humidity": "30-70%", "ph": "4.5-7.5", "rainfall": "90-200mm"},
    },
    "mothbeans": {
        "category": "Legume",
        "description": "A highly drought-resistant legume suited to arid and semi-arid regions with sandy soils.",
        "ideal_conditions": {"N": "0-40", "P": "35-60", "K": "15-25", "temperature": "24-32°C", "humidity": "40-65%", "ph": "3.5-9.9", "rainfall": "35-70mm"},
    },
    "mungbean": {
        "category": "Legume",
        "description": "A fast-growing, short-duration legume that improves soil fertility and prefers warm conditions.",
        "ideal_conditions": {"N": "0-40", "P": "35-60", "K": "15-25", "temperature": "27-30°C", "humidity": "80-90%", "ph": "6.2-7.2", "rainfall": "40-65mm"},
    },
    "blackgram": {
        "category": "Legume",
        "description": "A pulse crop grown in warm, humid conditions, often used in crop rotation for soil health.",
        "ideal_conditions": {"N": "20-60", "P": "55-80", "K": "15-25", "temperature": "25-35°C", "humidity": "60-75%", "ph": "6.5-7.5", "rainfall": "65-75mm"},
    },
    "lentil": {
        "category": "Legume",
        "description": "A cool-season pulse crop that grows well in loamy soils with moderate water needs.",
        "ideal_conditions": {"N": "0-40", "P": "60-80", "K": "15-25", "temperature": "18-30°C", "humidity": "60-70%", "ph": "6.0-7.0", "rainfall": "40-55mm"},
    },
    "pomegranate": {
        "category": "Fruit",
        "description": "A drought-tolerant fruit shrub that thrives in semi-arid climates with well-drained soil.",
        "ideal_conditions": {"N": "0-40", "P": "5-40", "K": "35-45", "temperature": "18-25°C", "humidity": "85-95%", "ph": "6.0-7.0", "rainfall": "100-110mm"},
    },
    "banana": {
        "category": "Fruit",
        "description": "A fast-growing tropical fruit crop requiring high nutrients, warmth, and consistent moisture.",
        "ideal_conditions": {"N": "80-120", "P": "70-95", "K": "45-55", "temperature": "25-30°C", "humidity": "75-85%", "ph": "5.5-6.5", "rainfall": "90-120mm"},
    },
    "mango": {
        "category": "Fruit",
        "description": "A tropical tree fruit tolerant of dry spells once established, needing warm temperatures.",
        "ideal_conditions": {"N": "0-40", "P": "15-40", "K": "25-35", "temperature": "27-37°C", "humidity": "45-55%", "ph": "4.5-7.0", "rainfall": "85-105mm"},
    },
    "grapes": {
        "category": "Fruit",
        "description": "A vine fruit favoring warm days, low humidity, and well-drained soil for quality yield.",
        "ideal_conditions": {"N": "0-40", "P": "120-145", "K": "195-205", "temperature": "8-25°C", "humidity": "80-84%", "ph": "5.5-6.5", "rainfall": "65-75mm"},
    },
    "watermelon": {
        "category": "Fruit",
        "description": "A warm-season fruit that needs plenty of sunshine, moderate humidity, and sandy loam soil.",
        "ideal_conditions": {"N": "80-120", "P": "5-25", "K": "45-55", "temperature": "24-27°C", "humidity": "80-90%", "ph": "6.0-7.0", "rainfall": "40-50mm"},
    },
    "muskmelon": {
        "category": "Fruit",
        "description": "A warm-climate fruit vine that needs well-drained soil and moderate irrigation.",
        "ideal_conditions": {"N": "80-120", "P": "5-25", "K": "45-55", "temperature": "27-30°C", "humidity": "90-95%", "ph": "6.0-7.0", "rainfall": "20-25mm"},
    },
    "apple": {
        "category": "Fruit",
        "description": "A temperate fruit tree requiring cool climates and chilling hours, unsuited to tropical heat.",
        "ideal_conditions": {"N": "0-40", "P": "120-145", "K": "195-205", "temperature": "21-24°C", "humidity": "90-95%", "ph": "5.5-6.5", "rainfall": "100-125mm"},
    },
    "orange": {
        "category": "Fruit",
        "description": "A citrus fruit tree that prefers subtropical climates with moderate rainfall and good drainage.",
        "ideal_conditions": {"N": "0-40", "P": "5-40", "K": "5-15", "temperature": "10-35°C", "humidity": "90-95%", "ph": "6.0-8.0", "rainfall": "100-120mm"},
    },
    "papaya": {
        "category": "Fruit",
        "description": "A fast-growing tropical fruit tree that needs warmth, sunlight, and well-drained fertile soil.",
        "ideal_conditions": {"N": "30-70", "P": "45-70", "K": "45-55", "temperature": "23-43°C", "humidity": "90-95%", "ph": "6.5-7.0", "rainfall": "40-250mm"},
    },
    "coconut": {
        "category": "Plantation",
        "description": "A coastal tropical palm requiring high humidity, warmth, and sandy or loamy soil.",
        "ideal_conditions": {"N": "0-40", "P": "5-40", "K": "25-35", "temperature": "25-30°C", "humidity": "90-100%", "ph": "5.5-6.5", "rainfall": "140-230mm"},
    },
    "cotton": {
        "category": "Fiber",
        "description": "A warm-season fiber crop requiring long frost-free periods and moderate to low rainfall.",
        "ideal_conditions": {"N": "100-140", "P": "35-60", "K": "15-25", "temperature": "22-26°C", "humidity": "75-85%", "ph": "5.5-7.0", "rainfall": "60-90mm"},
    },
    "jute": {
        "category": "Fiber",
        "description": "A fiber crop that thrives in hot, humid climates with high rainfall and fertile alluvial soil.",
        "ideal_conditions": {"N": "60-100", "P": "35-60", "K": "35-45", "temperature": "23-27°C", "humidity": "70-85%", "ph": "6.0-7.0", "rainfall": "150-200mm"},
    },
    "coffee": {
        "category": "Plantation",
        "description": "A tropical shrub grown at moderate elevations, needing shade, warmth, and consistent rainfall.",
        "ideal_conditions": {"N": "80-120", "P": "15-40", "K": "25-35", "temperature": "23-28°C", "humidity": "50-70%", "ph": "6.0-7.5", "rainfall": "150-200mm"},
    },
}


def get_crop_info(crop_name: str) -> dict:
    """Return the info dict for a crop (case-insensitive), or an empty dict."""
    return CROP_INFO.get(crop_name.lower().strip(), {})


def list_crops() -> list:
    """Return the sorted list of all crop names known to this module."""
    return sorted(CROP_INFO.keys())
