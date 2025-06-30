from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageDraw, ImageFont
import requests, base64, io
from bioclip_utils import reclassify_with_clip

app = FastAPI()

API_KEY = "YOUR_API_KEY"
MODEL_ID = "vegetable-classification-yekfv/1"  # Replace if different


def create_annotated_image(image, predictions):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for pred in predictions:
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        left, top = x - w/2, y - h/2
        right, bottom = x + w/2, y + h/2

        crop = image.crop((left, top, right, bottom))
        new_label, new_conf = reclassify_with_clip(crop)

        label = f"{new_label}: {new_conf:.2f}"
        draw.rectangle([left, top, right, bottom], outline="blue", width=3)

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([left, top - text_h - 10, left + text_w + 10, top], fill="blue")
        draw.text((left + 5, top - text_h - 5), label, fill="white", font=font)

    return image


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Prepare image for Roboflow
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    url = f"https://detect.roboflow.com/{MODEL_ID}"
    params = {"api_key": API_KEY, "confidence": 0.3, "overlap": 0.3, "format": "json"}

    response = requests.post(url, params=params, data=encoded,
                             headers={"Content-Type": "application/json"})

    if response.status_code != 200:
        return {"error": f"Roboflow API error: {response.status_code}"}

    result = response.json()
    predictions = result.get("predictions", [])

    if not predictions:
        return {"message": "No detections found."}

    annotated = create_annotated_image(image.copy(), predictions)
    buffered = io.BytesIO()
    annotated.save(buffered, format="JPEG")
    annotated_b64 = base64.b64encode(buffered.getvalue()).decode()

    return {
        "detections": predictions,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}"
    }
