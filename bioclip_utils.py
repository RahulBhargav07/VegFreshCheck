import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define your vegetable freshness classes
class_labels = [
    "Fresh Tomato", "Rotten Tomato",
    "Fresh Cabbage", "Rotten Cabbage"
]

def reclassify_with_clip(crop_img):
    image_input = preprocess(crop_img).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(c).to(device) for c in class_labels])

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        logits_per_image = image_features @ text_features.T
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    best_idx = int(probs.argmax())
    return class_labels[best_idx], float(probs[0][best_idx])
