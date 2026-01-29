import requests
import cv2
import json
import datetime
import os


API_URL = ""

IMAGE_PATH = r"C:\main\VScode\P4_anpr_project\P4_ANPR_final\images\photo_2026-01-12_15-50-14.jpg" 

print(f"Sending {IMAGE_PATH} to API...")

try:
    with open(IMAGE_PATH, "rb") as f:
        response = requests.post(API_URL, files={"file": f})
    

    if response.status_code == 200:
        result = response.json()
        print("\n API Response Received:")
        print(json.dumps(result, indent=2))
        img = cv2.imread(IMAGE_PATH)
        if img is not None:
            for item in result["results"]:
                box = item["box"] 
                text = item["text"]
                conf = item["confidence"]


                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                label = f"{text} ({conf:.2f})"
                cv2.putText(img, label, (box[0], box[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            output_folder = "results"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"result_{timestamp}.jpg"
            

            output_path = os.path.join(output_folder, output_filename)

            cv2.imwrite(output_path, img)
            print(f"\n Image saved automatically to: {output_path}")

    else:
        print(f" Error {response.status_code}: {response.text}")

except FileNotFoundError:
    print(f" Could not find image file: {IMAGE_PATH}")
except Exception as e:
    print(f" Connection failed: {e}")


