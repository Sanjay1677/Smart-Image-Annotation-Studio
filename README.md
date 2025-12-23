# Smart-Image-Annotation-Studio
Build a web-based tool for non-technical users to upload images, use bounding-box or point prompts to obtail segmentation masks(via SAM),review/refine results in a simple UII.
🧠 Smart Image Annotation Tool (SAM-based)

A web-based smart image annotation system built using Meta’s Segment Anything Model (SAM).
Users can interactively segment objects using point click or bounding box, preview results instantly, and download annotations in PNG and COCO JSON formats.
Annotation metadata is stored in SQLite for future retrieval.

🚀 Features

🖱️ Point-based segmentation (high accuracy, optimized mask selection)

📦 Bounding-box segmentation

🎯 Mask refinement (boundary clamping + morphological cleanup)

🖼️ Instant masked crop preview

⬇️ User-controlled downloads

Masked image (PNG)

COCO annotation (JSON)

🗃️ SQLite metadata storage (no image blobs)

🌐 React-based frontend

⚡ FastAPI backend

🔒 Stateless backend (no forced server storage)

🏗️ Tech Stack
Backend

Python 3.9+

FastAPI

Segment Anything Model (SAM – ViT-B)

OpenCV

NumPy

SQLite (Option A: metadata only)

Frontend

React.js

HTML5 Canvas

Axios

📁 Project Structure
project-root/
│
├── backend/
│   ├── app.py                  # FastAPI backend
│   ├── annotations.db          # SQLite DB (auto-created)
│   ├── sam_vit_b_01ec64.pth     # SAM checkpoint
│   └── uploads/                # Uploaded images (optional)
│
├── react-frontend/
│   ├── src/
│   │   ├── App.js
│   │   ├── ImageCanvas.js
│   │   ├── api.js
│   │   └── App.css
│   └── package.json
│
└── README.md

⚙️ Setup Instructions
1️⃣ Clone Repository
git clone <your-repo-url>
cd project-root

2️⃣ Backend Setup
Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

Install dependencies
pip install fastapi uvicorn torch torchvision
pip install opencv-python pillow numpy
pip install segment-anything

Download SAM checkpoint

Place this file in backend/:

sam_vit_b_01ec64.pth


Download from:
👉 https://github.com/facebookresearch/segment-anything

3️⃣ Run Backend
cd backend
uvicorn app:app --reload


Backend will run at:

http://127.0.0.1:8000

4️⃣ Frontend Setup
cd react-frontend
npm install
npm start


Frontend runs at:

http://localhost:3000

🖱️ How to Use

Upload an image

Click on an object (point segmentation)
OR
Drag a rectangle (box segmentation)

Preview the masked cropped object

Choose one:

⬇️ Download Mask (PNG)

⬇️ Download COCO (JSON)

Select save location via browser dialog

📦 COCO Output Format

Each segmentation produces a valid COCO-style JSON:

{
  "images": [
    {
      "id": "image_id",
      "width": 1024,
      "height": 768,
      "file_name": "image_id.png"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": "image_id",
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 12345,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "object"
    }
  ]
}


Compatible with:

CVAT

Label Studio

Detectron2

Custom training pipelines

🗃️ Database (SQLite – Option A)

Stored in annotations.db

Table: annotations
Column	Description
image_id	Image UUID
click_x	Click X coordinate
click_y	Click Y coordinate
bbox	Bounding box (JSON)
coco_json	COCO annotation (JSON)
created_at	Timestamp

❗ Images and masks are not stored in DB (lightweight by design)

🔒 Design Decisions

❌ No automatic server-side file saving

✅ User controls downloads

✅ Stateless backend

✅ Fast inference without retraining SAM

✅ Production-safe browser behavior

🧪 Known Limitations

Single-object per click

Category is fixed (object)

Polygon segmentation not enabled (bbox-only COCO)

🚀 Future Enhancements

COCO polygon segmentation (segmentation field)

Multi-class annotations

Undo / redo annotations

Annotation editor

ZIP export (PNG + COCO)

YOLO / Pascal VOC export

User authentication

📜 License

This project is for research and educational purposes.
SAM is subject to Meta AI’s license.
