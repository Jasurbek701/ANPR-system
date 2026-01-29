import io
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from typing import List, Dict


YOLO_MODEL_PATH = "models/best.onnx"
LPR_MODEL_PATH = "models/lprnet_uzbek_pro.onnx"
CHARS = "0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ"


IMG_SIZE = 640
CONF_THRES = 0.40
IOU_THRES = 0.50

class ANPRSystem:
    def __init__(self):

        print("Loading YOLO model...")
        self.yolo_session = ort.InferenceSession(YOLO_MODEL_PATH, providers=['CPUExecutionProvider'])
        self.yolo_input = self.yolo_session.get_inputs()[0].name
        
        print("Loading LPRNet model...")
        self.lpr_session = ort.InferenceSession(LPR_MODEL_PATH, providers=['CPUExecutionProvider'])
        self.lpr_input = self.lpr_session.get_inputs()[0].name
        print("Models loaded successfully.")


    def xywh2xyxy(self, x):
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def nms(self, boxes, scores, iou_threshold):
        if len(boxes) == 0: return []
        idxs = scores.argsort()[::-1]
        selected = []
        while len(idxs) > 0:
            current = idxs[0]
            selected.append(current)
            if len(idxs) == 1: break
            cur_box = boxes[current]
            other_boxes = boxes[idxs[1:]]
            x1 = np.maximum(cur_box[0], other_boxes[:, 0])
            y1 = np.maximum(cur_box[1], other_boxes[:, 1])
            x2 = np.minimum(cur_box[2], other_boxes[:, 2])
            y2 = np.minimum(cur_box[3], other_boxes[:, 3])
            inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            area_cur = (cur_box[2] - cur_box[0]) * (cur_box[3] - cur_box[1])
            area_oth = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
            union = area_cur + area_oth - inter
            iou = inter / (union + 1e-6)
            idxs = idxs[1:][iou < iou_threshold]
        return selected

    def preprocess_yolo(self, img):
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_input = img_resized[:, :, ::-1].transpose(2, 0, 1) 
        img_input = img_input.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        return img_input

    def decode_lpr(self, preds):
        pred_labels = []
        for i in range(preds.shape[0]):
            pred = preds[i]
            valid_chars = []
            for j in range(len(pred)):
                if pred[j] != len(CHARS) and (j == 0 or pred[j] != pred[j - 1]):
                    valid_chars.append(CHARS[pred[j]])
            pred_labels.append("".join(valid_chars))
        return pred_labels[0]

    def fix_uzbek_plate(self, text):
        # Your custom heuristic logic
        text = text.replace("I", "1")
        if len(text) < 7: return text
        region = text[:2]
        region = region.replace('O', '0').replace('Q', '0').replace('B', '8')
        clean = list(text)
        clean[0], clean[1] = region[0], region[1]
        is_type_1 = clean[2].isalpha() if len(clean) > 2 else False
        structure = ['D','D','L','D','D','D','L','L'] if is_type_1 else ['D','D','D','D','D','L','L','L']
        for i in range(min(len(clean), 8)):
            if i >= len(structure): break
            c, exp = clean[i], structure[i]
            if exp == 'D': 
                if c in ['O', 'Q']: clean[i] = '0'
                elif c == 'Z': clean[i] = '2'
                elif c == 'S': clean[i] = '5'
                elif c == 'B': clean[i] = '8'
                elif c == 'A': clean[i] = '4'
            elif exp == 'L': 
                if c == '0': clean[i] = 'O'
                elif c == '8': clean[i] = 'B'
                elif c == '5': clean[i] = 'S'
                elif c == '1': clean[i] = 'L'
        return "".join(clean)

    def recognize_plate(self, plate_img):
      
        img_resized = cv2.resize(plate_img, (128, 32))
        img_norm = (img_resized.astype(np.float32) / 127.5) - 1.0
        img_blob = np.transpose(img_norm, (2, 0, 1))
        img_blob = np.expand_dims(img_blob, axis=0)
        
        outputs = self.lpr_session.run(None, {self.lpr_input: img_blob})
        preds = np.argmax(outputs[0], axis=1)
        
    
        raw_text = self.decode_lpr(preds)
        final_text = self.fix_uzbek_plate(raw_text)
        return final_text

    def detect_and_recognize(self, img):
        h0, w0 = img.shape[:2]

        img_input = self.preprocess_yolo(img)
        outputs = self.yolo_session.run(None, {self.yolo_input: img_input})
        pred = outputs[0][0].T 

        conf = pred[:, 4]
        mask = conf > CONF_THRES
        pred = pred[mask]
        conf = conf[mask]

        if len(pred) == 0:
            return []


        boxes_xywh = pred[:, :4]
        boxes_xyxy = self.xywh2xyxy(boxes_xywh)
        

        gain_w = w0 / IMG_SIZE
        gain_h = h0 / IMG_SIZE
        boxes_xyxy[:, [0, 2]] *= gain_w
        boxes_xyxy[:, [1, 3]] *= gain_h
        boxes_xyxy = boxes_xyxy.astype(int)


        keep = self.nms(boxes_xyxy, conf, IOU_THRES)
        boxes_xyxy = boxes_xyxy[keep]
        conf = conf[keep]

        results = []
        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w0, x2), min(h0, y2)
            

            plate_crop = img[y1:y2, x1:x2]
            if plate_crop.size == 0: continue

            text = self.recognize_plate(plate_crop)
            
            results.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "text": text,
                "confidence": float(conf[i])
            })
            
        return results


anpr_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global anpr_system
    anpr_system = ANPRSystem()
    yield


app = FastAPI(lifespan=lifespan)

@app.post("/detect")
async def detect_plate(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")


        detections = anpr_system.detect_and_recognize(img)
        
        return {
            "filename": file.filename,
            "plate_count": len(detections),
            "results": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))