import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# Load SAM model
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading SAM model...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
print("Model loaded successfully!")

# Load image
image_path = "images/sanjay.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

predictor.set_image(image_rgb)

# Variables for box drawing
drawing_box = False
start_point = None
end_point = None
temp_image = image_bgr.copy()

# Threshold to differentiate between click vs drag
DRAG_THRESHOLD = 5


# -------------------------------------------------------
# POINT SEGMENTATION (your old function — unchanged)
# -------------------------------------------------------
def segment_point_and_show(point):
    plt.close('all')

    input_point = np.array([point])
    input_label = np.array([1])  # 1 = foreground

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(6, 6))
        plt.imshow(image_rgb)
        plt.imshow(mask, alpha=0.5)
        plt.title(f"Point Mask {i+1} - Score: {score:.4f}")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.001)


# -------------------------------------------------------
# BOX SEGMENTATION (new function)
# -------------------------------------------------------
def segment_box_and_show(box):
    plt.close('all')

    input_box = np.array(box)

    masks, scores, logits = predictor.predict(
        box=input_box,
        multimask_output=False
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(6, 6))
        plt.imshow(image_rgb)
        plt.imshow(mask, alpha=0.5)
        plt.title(f"Box Mask {i+1} - Score: {score:.4f}")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.001)


# -------------------------------------------------------
# MOUSE EVENT HANDLER — decides point vs box
# -------------------------------------------------------
def mouse_event(event, x, y, flags, param):
    global drawing_box, start_point, end_point, temp_image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_box = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing_box:
        temp_image = image_bgr.copy()
        end_point = (x, y)
        cv2.rectangle(temp_image, start_point, end_point, (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing_box = False
        end_point = (x, y)

        dx = abs(end_point[0] - start_point[0])
        dy = abs(end_point[1] - start_point[1])

        # If tiny movement → treat as point click
        if dx < DRAG_THRESHOLD and dy < DRAG_THRESHOLD:
            print("Point selected:", start_point)
            segment_point_and_show(start_point)
            return

        # Else treat as bounding box
        x1, y1 = start_point
        x2, y2 = end_point
        box = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]

        print("Box selected:", box)
        segment_box_and_show(box)


# Main window
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_event)

while True:
    cv2.imshow("Image", temp_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
