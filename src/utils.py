import cv2
import numpy as np

def extract_color_histogram(frame, bbox, bins=(8, 8, 8)):
    """
    Extracts a 3D color histogram from the HSV color space for a given bounding box.

    Args:
        frame (np.ndarray): The video frame.
        bbox (list or tuple): The bounding box [x1, y1, x2, y2].
        bins (tuple): The number of bins for each channel (H, S, V).

    Returns:
        np.ndarray: A flattened, normalized color histogram.
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        player_crop = frame[y1:y2, x1:x2]
        
        if player_crop.size == 0:
            return None

        # Convert to HSV color space, which is better for handling lighting changes
        hsv_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram
        hist = cv2.calcHist([hsv_crop], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        
        # Normalize the histogram
        cv2.normalize(hist, hist)
        
        # Return the flattened histogram as a feature vector
        return hist.flatten()
    except Exception as e:
        print(f"Error extracting histogram: {e}")
        return None