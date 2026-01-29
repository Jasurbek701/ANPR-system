import requests
import cv2
import json
import datetime
import os

# 1. SETUP
API_URL = "http://localhost:8000/detect"
# Use 'r' before the string to handle backslashes correctly in Windows paths
IMAGE_PATH = r"C:\main\VScode\P4_anpr_project\P4_ANPR_final\images\photo_2026-01-12_15-50-14.jpg" 

print(f"Sending {IMAGE_PATH} to API...")

# 2. SEND REQUEST
try:
    with open(IMAGE_PATH, "rb") as f:
        response = requests.post(API_URL, files={"file": f})
    
    # 3. CHECK RESPONSE
    if response.status_code == 200:
        result = response.json()
        print("\n API Response Received:")
        print(json.dumps(result, indent=2))
        
        # 4. DRAW RESULTS
        img = cv2.imread(IMAGE_PATH)
        if img is not None:
            for item in result["results"]:
                box = item["box"]  # [x1, y1, x2, y2]
                text = item["text"]
                conf = item["confidence"]

                # Draw Rectangle
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                # Draw Text
                label = f"{text} ({conf:.2f})"
                cv2.putText(img, label, (box[0], box[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # --- NEW SAVING LOGIC ---
            
            # A. Create a 'results' folder if it doesn't exist
            output_folder = "results"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # B. Generate automatic filename with timestamp
            # Format: result_YYYYMMDD_HHMMSS.jpg
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"result_{timestamp}.jpg"
            
            # C. Combine folder and filename
            output_path = os.path.join(output_folder, output_filename)

            # D. Save the image locally
            cv2.imwrite(output_path, img)
            print(f"\n Image saved automatically to: {output_path}")

    else:
        print(f" Error {response.status_code}: {response.text}")

except FileNotFoundError:
    print(f" Could not find image file: {IMAGE_PATH}")
except Exception as e:
    print(f" Connection failed: {e}")
