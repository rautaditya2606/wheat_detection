import cv2
import numpy as np
import os

def apply_color_mask(image, lower_thresh, upper_thresh):
    """Applies a color mask and returns the filtered regions."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask

def draw_highlight(image, mask, color=(0, 0, 255), label=None):
    """Draws contours around the mask on the image."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 20: # Filter noise
            cv2.drawContours(output, [cnt], -1, color, 2)
    
    if label:
        cv2.putText(output, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return output

def get_leaf_mask(image):
    """Detects the leaf using edge detection and returns a mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    leaf_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(leaf_mask, [largest_cnt], -1, 255, -1)
    return leaf_mask

# --- Heuristic Functions for 15 Diseases ---

def highlight_black_rust(image):
    # Elongated, dark reddish-brown to black pustules on stems/sheaths
    lower_black = np.array([0, 50, 20])
    upper_black = np.array([20, 255, 100])
    
    lower_brown = np.array([5, 100, 50])
    upper_brown = np.array([15, 255, 200])

    _, mask_black = apply_color_mask(image, lower_black, upper_black)
    _, mask_brown = apply_color_mask(image, lower_brown, upper_brown)
    
    combined_mask = cv2.bitwise_or(mask_black, mask_brown)
    return draw_highlight(image, combined_mask, color=(0, 0, 0), label="Black Rust (with brown spots)")

def highlight_yellow_rust(image):
    # Bright yellow pustules in linear rows (stripes)
    lower = np.array([20, 100, 100])
    upper = np.array([35, 255, 255])
    _, mask = apply_color_mask(image, lower, upper)
    return draw_highlight(image, mask, color=(0, 255, 255), label="Yellow Rust")

def highlight_brown_rust(image):
    # Small, orange-brown pustules scattered irregularly
    leaf_mask = get_leaf_mask(image)
    
    # Precise HSV range for reddish-brown spots based on uploaded image
    # The spots are quite "rusty" (Hue: 0-20, Sat: 50-255, Val: 40-200)
    lower = np.array([0, 50, 40])
    upper = np.array([20, 255, 200])
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, lower, upper)
    
    # Increased closing kernel to join the dense scattered spots seen in the image
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Remove very tiny noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open)
    
    # CRITICAL: Intersection with leaf mask to ONLY highlight spots on the leaf
    combined_mask = cv2.bitwise_and(color_mask, leaf_mask)
    
    return draw_highlight(image, combined_mask, color=(0, 69, 139), label="Brown Rust")

def highlight_fusarium_head_blight(image):
    # Bleached or tan spikelets, pink/orange/yellowish-brown fungal growth
    leaf_mask = get_leaf_mask(image)
    
    # Wider range to include tan, pinkish, and yellowish-brown
    lower1 = np.array([0, 30, 100])  # Pinkish/Reddish
    upper1 = np.array([20, 255, 255])
    
    lower2 = np.array([20, 40, 100]) # Yellowish/Tan
    upper2 = np.array([40, 255, 255])
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    
    color_mask = cv2.bitwise_or(mask1, mask2)
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    
    combined_mask = cv2.bitwise_and(color_mask, leaf_mask)
    
    return draw_highlight(image, combined_mask, color=(0, 0, 139), label="Fusarium Head Blight")

def highlight_wheat_blast(image):
    # Sudden bleaching, eye-shaped lesion on stem
    lower = np.array([0, 0, 180])
    upper = np.array([180, 30, 255])
    _, mask = apply_color_mask(image, lower, upper)
    return draw_highlight(image, mask, color=(255, 255, 255), label="Wheat Blast")

def highlight_smut(image):
    # Dusty mass of olive-black spores
    # Focusing on the dark-reddish-brownish part
    leaf_mask = get_leaf_mask(image)
    lower = np.array([0, 50, 20])
    upper = np.array([20, 255, 150])
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, lower, upper)
    combined_mask = cv2.bitwise_and(color_mask, leaf_mask)
    
    return draw_highlight(image, combined_mask, color=(50, 50, 50), label="Smut")

def highlight_powdery_mildew(image):
    # White to greyish-white powdery coating
    leaf_mask = get_leaf_mask(image)
    lower = np.array([0, 0, 200])
    upper = np.array([180, 40, 255])
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, lower, upper)
    combined_mask = cv2.bitwise_and(color_mask, leaf_mask)
    
    return draw_highlight(image, combined_mask, color=(255, 255, 255), label="Powdery Mildew")

def highlight_septoria(image):
    # Tan, lens-shaped lesions with yellow halo and tiny black specks
    leaf_mask = get_leaf_mask(image)
    lower = np.array([10, 50, 50])
    upper = np.array([30, 255, 200])
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, lower, upper)
    combined_mask = cv2.bitwise_and(color_mask, leaf_mask)
    
    return draw_highlight(image, combined_mask, color=(0, 255, 0), label="Septoria")

def highlight_tan_spot(image):
    # Oval-shaped, tan lesions with dark brown center and yellow halo
    lower = np.array([10, 40, 40])
    upper = np.array([25, 255, 255])
    _, mask = apply_color_mask(image, lower, upper)
    return draw_highlight(image, mask, color=(50, 100, 150), label="Tan Spot")

def highlight_common_root_rot(image):
    # Dark brown to black discolouration and decay at base
    lower = np.array([0, 0, 0])
    upper = np.array([20, 255, 80])
    _, mask = apply_color_mask(image, lower, upper)
    return draw_highlight(image, mask, color=(40, 40, 40), label="Common Root Rot")

def highlight_aphid(image):
    # Small, green or brown insects
    lower = np.array([30, 50, 50])
    upper = np.array([90, 255, 255]) # Green aphids
    _, mask = apply_color_mask(image, lower, upper)
    return draw_highlight(image, mask, color=(0, 100, 0), label="Aphids")

def highlight_mite(image):
    # Stippling (tiny white or yellow spots)
    # Focusing on light green spots
    leaf_mask = get_leaf_mask(image)
    lower_light_green = np.array([35, 40, 100])
    upper_light_green = np.array([85, 255, 255])
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, lower_light_green, upper_light_green)
    combined_mask = cv2.bitwise_and(color_mask, leaf_mask)
    
    return draw_highlight(image, combined_mask, color=(255, 255, 0), label="Mites")

def highlight_stem_fly(image):
    # Central leaf withering (dead heart), tunneling
    lower = np.array([0, 0, 100])
    upper = np.array([180, 30, 200]) # Example for withered/dried tissue
    _, mask = apply_color_mask(image, lower, upper)
    return draw_highlight(image, mask, color=(128, 128, 128), label="Stem Fly")

def highlight_leaf_blight(image):
    leaf_mask = get_leaf_mask(image)
    lower_yellow_brown = np.array([10, 50, 20])
    upper_yellow_brown = np.array([30, 255, 200])
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, lower_yellow_brown, upper_yellow_brown)
    combined_mask = cv2.bitwise_and(color_mask, leaf_mask)
    
    return draw_highlight(image, combined_mask, color=(0, 50, 100), label="Leaf Blight")

def highlight_healthy(image):
    return image

# --- Dispatcher ---

DISPATCHER = {
    "Black Rust": highlight_black_rust,
    "Yellow Rust": highlight_yellow_rust,
    "Brown Rust": highlight_brown_rust,
    "Blast": highlight_wheat_blast,
    "Fusarium Head Blight": highlight_fusarium_head_blight,
    "Smut": highlight_smut,
    "Mildew": highlight_powdery_mildew,
    "Septoria": highlight_septoria,
    "Tan spot": highlight_tan_spot,
    "Common Root Rot": highlight_common_root_rot,
    "Aphid": highlight_aphid,
    "Mite": highlight_mite,
    "Stem fly": highlight_stem_fly,
    "Leaf Blight": highlight_leaf_blight,
    "Healthy": highlight_healthy
}

def highlight_infection(image_path, predicted_class, output_path):
    """
    Main entry point to apply highlighting based on prediction.
    """
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    if predicted_class in DISPATCHER:
        processed_image = DISPATCHER[predicted_class](image)
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, processed_image)
        return True
    else:
        return False
