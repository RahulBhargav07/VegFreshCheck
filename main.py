from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import os
from typing import Dict, Any

# === FastAPI App ===
app = FastAPI(
    title="Vegetable Freshness Detection API",
    version="1.0.0",
    description="Detects and classifies vegetables as fresh or rotten using a Roboflow detection model."
)

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Detection Service ===
class VegetableDetectionService:
    def __init__(self):
        self.api_key = os.getenv("ROBOFLOW_API_KEY", "NVfp8h9atJEAWzsw1eZ0")  # Replace if needed
        self.model_id = "vegetable-classification-yekfv/4"
        self.base_url = "https://detect.roboflow.com"

    def create_annotated_image(self, image: Image.Image, predictions: list) -> str:
        draw = ImageDraw.Draw(image)

        try:
            font_paths = [
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ]
            font = None
            for path in font_paths:
                try:
                    font = ImageFont.truetype(path, 20)
                    break
                except:
                    continue
            if not font:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        for pred in predictions:
            cx, cy, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            left, top = cx - w / 2, cy - h / 2
            right, bottom = cx + w / 2, cy + h / 2

            draw.rectangle([left, top, right, bottom], outline="green", width=3)
            label = f"{pred['class']}: {pred['confidence']:.2f}"
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([left, top - th - 10, left + tw + 10, top], fill="green")
            draw.text((left + 5, top - th - 5), label, fill="white", font=font)

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode()

    async def detect_vegetables(self, file: UploadFile) -> Dict[str, Any]:
        try:
            image_bytes = await file.read()
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")

            url = f"{self.base_url}/{self.model_id}"
            params = {
                "api_key": self.api_key,
                "confidence": 0.3,
                "overlap": 0.3,
                "format": "json"
            }

            response = requests.post(
                url,
                params=params,
                data=img_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30
            )

            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Roboflow API error: {response.text}")

            result = response.json()
            predictions = result.get("predictions", [])

            annotated_b64 = None
            if predictions:
                img = Image.open(BytesIO(image_bytes))
                annotated_b64 = self.create_annotated_image(img, predictions)

            return {
                "success": True,
                "detection_count": len(predictions),
                "predictions": [
                    {
                        "class": pred["class"],
                        "confidence": round(pred["confidence"], 4),
                        "position": {"x": pred["x"], "y": pred["y"]},
                        "size": {"width": pred["width"], "height": pred["height"]},
                        "detection_id": pred.get("detection_id")
                    } for pred in predictions
                ],
                "image_info": result.get("image", {}),
                "annotated_image": annotated_b64,
                "processing_time": result.get("time", 0)
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# === Initialize Service ===
veg_service = VegetableDetectionService()

# === Routes ===
@app.get("/")
async def root():
    return {
        "message": "Vegetable Freshness Detection API is live ✅",
        "status": "running",
        "endpoints": {
            "detect": "/detect (POST)",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/detect")
async def detect_vegetable(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be an image.")

    if hasattr(file, "size") and file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Max size is 10MB.")

    result = await veg_service.detect_vegetables(file)
    return JSONResponse(content=result)
