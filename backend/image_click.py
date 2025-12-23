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

# Set image once for predictor
predictor.set_image(image_rgb)

clicked_point = []

# Enable interactive mode for matplotlib
plt.ion()

# Mouse click handler
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point.clear()
        clicked_point.append((x, y))
        print(f"Clicked at: {x}, {y}")
        segment_and_show(image_rgb, clicked_point[0])

# Run SAM segmentation and show result
def segment_and_show(image_rgb, point):
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
        plt.title(f"Mask {i+1} - Score: {score:.4f}")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.001)

# OpenCV window to capture mouse clicks
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", click_event)

while True:
    cv2.imshow("Image", image_bgr)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
