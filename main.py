from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import base64
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

app = FastAPI()

# Replace with your actual Roboflow API key and model ID
ROBOFLOW_API_KEY = "YOUR_API_KEY"
MODEL_ID = "vegetable-classification-yekfv/1"

# Optional: Resize image before sending (faster & less memory)
def resize_image(image: Image.Image, max_size=(512, 512)):
    image.thumbnail(max_size)
    return image

def annotate_image(image: Image.Image, predictions):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for pred in predictions:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        left, top = x - w / 2, y - h / 2
        right, bottom = x + w / 2, y + h / 2
        label = f"{pred['class']}: {pred['confidence']:.2f}"

        draw.rectangle([left, top, right, bottom], outline="green", width=2)
        text_w, text_h = draw.textsize(label, font=font)
        draw.rectangle([left, top - text_h - 4, left + text_w + 4, top], fill="green")
        draw.text((left + 2, top - text_h - 2), label, fill="white", font=font)

    return image

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        image = resize_image(image)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = requests.post(
            f"https://detect.roboflow.com/{MODEL_ID}",
            params={
                "api_key": ROBOFLOW_API_KEY,
                "confidence": 0.4,
                "overlap": 0.3,
                "format": "json"
            },
            data=img_str,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        if response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": response.text})

        result = response.json()
        predictions = result.get("predictions", [])

        # Annotate and save image
        annotated_image = annotate_image(image.copy(), predictions)
        annotated_buffer = BytesIO()
        annotated_image.save(annotated_buffer, format="JPEG")
        annotated_b64 = base64.b64encode(annotated_buffer.getvalue()).decode("utf-8")

        return {
            "detections": predictions,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
