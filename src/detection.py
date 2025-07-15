import torch
import numpy as np
import cv2
import os
import requests
from tqdm import tqdm

from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression, scale_boxes

_model_cache = {}
MODEL_URL = "https://huggingface.co/manankadel/player-reid-yolov8/resolve/main/best.pt?download=true"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")

def download_model_if_needed(url, path):
    if os.path.exists(path):
        print(f"Model already exists at {path}. Skipping download.")
        return
    
    print(f"Model not found. Downloading from {url} to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(path, 'wb') as f, tqdm(
            desc=os.path.basename(path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                f.write(data)
                
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        if os.path.exists(path):
            os.remove(path)
        raise

def load_model_once(model_path, device):
    global _model_cache
    if model_path not in _model_cache:
        download_model_if_needed(MODEL_URL, model_path)
        
        print(f"Loading model from {model_path} into memory...")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model = ckpt['model'].float().eval()
        _model_cache[model_path] = model.to(device)
    else:
        print("Loading model from cache.")
        
    return _model_cache[model_path]

# --- THE FIX IS HERE ---
# The function signature now correctly accepts model_path
def detect_players(frame, model_path, conf_threshold=0.4, iou_threshold=0.5):
    """Detects players using the model, which is downloaded on first run."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # We use the passed model_path to load the model
    model = load_model_once(model_path, device)
    
    # Pre-processing
    imgsz = 640
    stride = int(model.stride.max())
    letterbox = LetterBox(new_shape=(imgsz, imgsz), stride=stride, auto=True)
    img = letterbox(image=frame)
    
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        preds = model(img)[0]

    # Post-processing
    results = non_max_suppression(preds, conf_thres=conf_threshold, iou_thres=iou_threshold)
    
    detections = []
    det = results[0]
    if det is not None and len(det):
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
        for *xyxy, conf, cls in det:
            detections.append([
                xyxy[0].item(), xyxy[1].item(), 
                xyxy[2].item(), xyxy[3].item(), 
                conf.item(), cls.item()
            ])
            
    return detections