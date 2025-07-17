import cv2
import numpy as np
import torch
from PIL import Image

def load_image_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Press SPACE to capture the image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image from camera.")

        cv2.imshow("Press SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE key
            captured_frame = frame
            break
        elif key == 27:  # ESC key to exit
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("Image capture canceled.")

    cap.release()
    cv2.destroyAllWindows()
    return Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))

def convert_to_numpy(image):
    return np.array(image)

def detect_body_yolov5(image_np):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    results = model(image_np)
    return results

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def extract_person(image_np, results):
    height, width = image_np.shape[:2]
    for result in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, confidence, cls = result
        if int(cls) == 0:  # '0' corresponds to 'person' class in YOLOv5
            x1 = clamp(int(x1), 0, width - 1)
            y1 = clamp(int(y1), 0, height - 1)
            x2 = clamp(int(x2), 0, width - 1)
            y2 = clamp(int(y2), 0, height - 1)

            person_crop = image_np[y1:y2, x1:x2]
            mask = np.zeros_like(image_np[:, :, 0])
            mask[y1:y2, x1:x2] = 255
            return person_crop, mask, (x1, y1)
    return None, None, None

def blend_images(image1_np, person_crop, mask, position):
    mask = mask[position[1]:position[1]+person_crop.shape[0],
                position[0]:position[0]+person_crop.shape[1]]

    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if person_crop.shape[:2] != mask.shape[:2]:
        raise RuntimeError("Mask and person_crop must be the same size.")

    center = (position[0] + person_crop.shape[1] // 2,
              position[1] + person_crop.shape[0] // 2)

    blended_image = cv2.seamlessClone(person_crop, image1_np, mask, center,
                                      cv2.NORMAL_CLONE)

    return blended_image

def main():
    output_path = 'combined_image.jpg'

    print("Capturing first image...")
    image1 = load_image_from_camera()

    print("Capturing second image...")
    image2 = load_image_from_camera()

    image1_np = convert_to_numpy(image1)
    image2_np = convert_to_numpy(image2)

    results = detect_body_yolov5(image2_np)

    person_crop, mask, position = extract_person(image2_np, results)
    if person_crop is None:
        raise RuntimeError("No person detected in the second image.")

    combined_image = blend_images(image1_np, person_crop, mask, position)

    Image.fromarray(combined_image).save(output_path)
    print(f"Combined image saved to {output_path}")

if __name__ == "__main__":
    main()
