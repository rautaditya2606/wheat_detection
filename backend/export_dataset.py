import os
import sys
import requests
import argparse
from PIL import Image
from io import BytesIO

# Add backend directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app import app
from models import db, Feedback

# Map to ensure class names exactly match the directory structure of the training set
CLASS_NAME_CLEANUP = {
    "aphid": "Aphid",
    "black_rust": "Black Rust",
    "blast": "Blast",
    "brown_rust": "Brown Rust",
    "common_root_rot": "Common Root Rot",
    "fusarium_head_blight": "Fusarium Head Blight",
    "healthy": "Healthy",
    "leaf_blight": "Leaf Blight",
    "mildew": "Mildew",
    "mite": "Mite",
    "septoria": "Septoria",
    "smut": "Smut",
    "stem_fly": "Stem fly",
    "tan_spot": "Tan spot",
    "yellow_rust": "Yellow Rust",
}

def export_dataset(limit=None):
    dataset_dir = os.path.join(current_dir, "data", "retrain_dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    with app.app_context():
        # Fetch verified feedback not yet used in training
        query = Feedback.query.filter_by(is_verified=True, used_in_training=False)
        if limit:
            query = query.limit(limit)
            
        records = query.all()
        if not records:
            print("No new verified feedback records found for export.")
            return 0
            
        print(f"Found {len(records)} verified records to export...")
        
        exported_count = 0
        for f in records:
            # 1. Determine correct class
            raw_class = f.correct_class if (f.correct_class and not f.is_correct) else f.predicted_class
            
            # Map/normalize to target folder name case
            class_folder_name = CLASS_NAME_CLEANUP.get(raw_class.lower().replace(" ", "_"), raw_class)
            
            # 2. Setup output folder
            class_dir = os.path.join(dataset_dir, class_folder_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Target image path
            img_path = os.path.join(class_dir, f"{f.id}.jpg")
            
            # 3. Download / copy image
            try:
                if f.image_url.startswith("http"):
                    # Download from Cloudinary
                    resp = requests.get(f.image_url, timeout=15)
                    if resp.status_code == 200:
                        image = Image.open(BytesIO(resp.content)).convert("RGB")
                        image.save(img_path, "JPEG")
                    else:
                        print(f"Failed to download image from {f.image_url}: HTTP {resp.status_code}")
                        continue
                else:
                    # Copy local file
                    local_src = os.path.join(current_dir, "static", "uploads", f.image_url)
                    if not os.path.exists(local_src):
                        local_src = os.path.join(current_dir, f.image_url)
                        
                    if os.path.exists(local_src):
                        image = Image.open(local_src).convert("RGB")
                        image.save(img_path, "JPEG")
                    else:
                        print(f"Local image source not found: {local_src}")
                        continue
                        
                # 4. Mark as used in training
                f.used_in_training = True
                exported_count += 1
                
            except Exception as e:
                print(f"Error exporting feedback {f.id}: {e}")
                
        # Commit session changes
        if exported_count > 0:
            db.session.commit()
            print(f"Successfully exported {exported_count} images to {dataset_dir} and marked them in DB.")
        
        return exported_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export verified user feedback for model retraining.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of exported images.")
    args = parser.parse_args()
    
    export_dataset(limit=args.limit)
