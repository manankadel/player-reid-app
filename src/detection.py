import torch
import numpy as np
import cv2

# Import the necessary pre-processing and post-processing functions
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression, scale_boxes

# A dictionary to cache the loaded model
_model_cache = {}

def load_model_once(model_path, device):
    """Loads the model and caches it."""
    global _model_cache
    if model_path not in _model_cache:
        print(f"Loading model from {model_path}...")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model = ckpt['model'].float().eval()
        _model_cache[model_path] = model.to(device)
    return _model_cache[model_path]

def detect_players(frame, model_path, conf_threshold=0.4, iou_threshold=0.5):
    """
    Detects players in a given frame using a manual pipeline that includes
    the critical letterboxing pre-processing step.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- 1. Load Model ---
    model = load_model_once(model_path, device)
    
    # --- 2. Pre-process Frame: Letterboxing ---
    # The model expects a fixed-size input, typically 640x640.
    # Letterboxing resizes the image while maintaining aspect ratio and pads it.
    imgsz = 640
    stride = int(model.stride.max()) # Get model's stride
    
    # Create the letterbox transformer
    letterbox = LetterBox(new_shape=(imgsz, imgsz), stride=stride, auto=True)
    
    # Apply letterboxing to the frame
    img = letterbox(image=frame)
    
    # Convert to CHW format, to tensor, and normalize
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # --- 3. Inference ---
    with torch.no_grad():
        preds = model(img)[0]

    # --- 4. Post-process: Non-Max Suppression ---
    results = non_max_suppression(
        preds, 
        conf_thres=conf_threshold, 
        iou_thres=iou_threshold
    )
    
    # --- 5. Format Detections ---
    detections = []
    det = results[0]
    if det is not None and len(det):
        # Rescale boxes from the letterboxed image size back to the original frame size
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
        
        for *xyxy, conf, cls in det:
            detections.append([
                xyxy[0].item(), xyxy[1].item(), 
                xyxy[2].item(), xyxy[3].item(), 
                conf.item(), cls.item()
            ])
            
    return detections