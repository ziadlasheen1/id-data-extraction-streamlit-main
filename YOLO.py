from ultralytics import YOLO
import cv2
def detect_and_crop(frame):
    model = YOLO("best.pt")  # use your trained weights
    results = model.predict(frame)
    crops = []
    box_dict = {}  # class_name: crop_path

    all_found = True
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]

            if conf < 0.8:
                all_found = False
                continue
            
            # Save cropped image
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            path = f"temp_crops/{class_name}.jpg"
            cv2.imwrite(path, crop)
            box_dict[class_name] = path

    # Check if all 8 expected classes are in box_dict
    expected_classes = {"name", "number", "Code", "family", "state", "image", "city", "neighborhood"}
    all_boxes_found = expected_classes.issubset(set(box_dict.keys()))

    return box_dict, all_boxes_found
