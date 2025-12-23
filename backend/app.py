import os
import uuid
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
import torch
import sqlite3
from fastapi.middleware.cors import CORSMiddleware
from segment_anything import sam_model_registry, SamPredictor
import base64
# -------------------------
# SQLite Setup
# -------------------------
DB_PATH = "sam.db"



def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT,
            click_x INTEGER,
            click_y INTEGER,
            bbox TEXT,
            coco_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

import json

def save_annotation(image_id, x, y, bbox, coco_json):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO annotations (image_id, click_x, click_y, bbox, coco_json)
        VALUES (?, ?, ?, ?, ?)
    """, (
        image_id,
        x,
        y,
        json.dumps(bbox),
        json.dumps(coco_json)
    ))
    conn.commit()
    conn.close()


# -------------------------
# Folders
# -------------------------
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# SAM Model Load
# -------------------------
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading SAM model...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)
print("SAM loaded on:", DEVICE)

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="Optimized SAM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

image_registry = {}

def mask_to_coco(mask, image_id, category_id=1):
    ys, xs = np.where(mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    area = int(mask.sum())

    return {
        "images": [
            {
                "id": image_id,
                "width": int(mask.shape[1]),
                "height": int(mask.shape[0]),
                "file_name": f"{image_id}.png"
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x_min, y_min, width, height],
                "area": area,
                "iscrowd": 0
            }
        ],
        "categories": [
            {
                "id": category_id,
                "name": "object"
            }
        ]
    }


# -------------------------
# Utilities
# -------------------------
def read_image_bytes_to_rgb(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))

def save_raw_mask(image_rgb, mask, save_path):
    mask_uint8 = (mask * 255).astype(np.uint8)
    pil_mask = Image.fromarray(mask_uint8)

    rgba = Image.fromarray(image_rgb).convert("RGBA")
    rgba.putalpha(pil_mask)

    bbox = pil_mask.getbbox()
    if bbox is None:
        return None

    cropped = rgba.crop(bbox)
    cropped.save(save_path)
    return cropped

# -------------------------
# Request Models
# -------------------------
class PointSegRequest(BaseModel):
    image_id: str
    x: int
    y: int

class BoxSegRequest(BaseModel):
    image_id: str
    x1: int
    y1: int
    x2: int
    y2: int

# -------------------------
# Upload
# -------------------------
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    data = await file.read()
    image_rgb = read_image_bytes_to_rgb(data)

    image_id = str(uuid.uuid4())
    with open(os.path.join(UPLOAD_DIR, f"{image_id}.png"), "wb") as f:
        f.write(data)

    image_registry[image_id] = {"rgb": image_rgb}
    return {"image_id": image_id}

# -------------------------
# OPTIMIZED POINT SEGMENTATION
# -------------------------
@app.post("/segment/point")
async def segment_point(req: PointSegRequest):

    meta = image_registry.get(req.image_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Image not found")

    image_rgb = meta["rgb"]
    predictor.set_image(image_rgb)

    point = np.array([[req.x, req.y]])
    label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=point,
        point_labels=label,
        multimask_output=True
    )

    def score_mask(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return -1
        area = len(xs)
        cx, cy = xs.mean(), ys.mean()
        dist = np.sqrt((cx - req.x) ** 2 + (cy - req.y) ** 2)
        return -area - dist * 10

    best_idx = max(range(len(masks)), key=lambda i: score_mask(masks[i]))
    mask = masks[best_idx]

    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise HTTPException(status_code=400, detail="Empty mask")

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    clamped = np.zeros_like(mask)
    clamped[y1:y2 + 1, x1:x2 + 1] = mask[y1:y2 + 1, x1:x2 + 1]
    mask = clamped

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = mask.astype(bool)

    # 🔥 CREATE CROPPED IMAGE (IN MEMORY ONLY)
    mask_uint8 = (mask * 255).astype(np.uint8)
    pil_mask = Image.fromarray(mask_uint8)

    rgba = Image.fromarray(image_rgb).convert("RGBA")
    rgba.putalpha(pil_mask)

    bbox = pil_mask.getbbox()
    if bbox is None:
        raise HTTPException(status_code=400, detail="Empty crop")

    cropped = rgba.crop(bbox)

    import base64
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")

    coco_json = mask_to_coco(mask, req.image_id)
    
    # 🔐 SAVE METADATA TO SQLITE (OPTION A)
    save_annotation(
        image_id=req.image_id,
        x=req.x,
        y=req.y,
        bbox=[int(x1), int(y1), int(x2), int(y2)],
        coco_json=coco_json
    )


    return {
    "raw_mask_base64": base64.b64encode(buf.getvalue()).decode(),
    "filename": f"mask_{req.image_id}_{req.x}_{req.y}.png",
    "coco": coco_json
}


# -------------------------
# BOX SEGMENTATION (UNCHANGED)
# -------------------------
@app.post("/segment/box")
async def segment_box(req: BoxSegRequest):

    meta = image_registry.get(req.image_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Image not found")

    image_rgb = meta["rgb"]
    predictor.set_image(image_rgb)

    box = np.array([req.x1, req.y1, req.x2, req.y2])
    masks, _, _ = predictor.predict(box=box, multimask_output=False)
    mask = masks[0]

    raw_path = os.path.join(
        OUTPUT_DIR,
        f"{req.image_id}_box_{req.x1}_{req.y1}_{req.x2}_{req.y2}_raw.png"
    )

    cropped = save_raw_mask(image_rgb, mask, raw_path)

    import base64
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")

    return {"raw_mask_base64": base64.b64encode(buf.getvalue()).decode()}

# -------------------------
# Root
# -------------------------
@app.get("/")
def index():
    return HTMLResponse("<h2>SAM Optimized API Running</h2>")

@app.get("/annotations/{image_id}")
def get_annotations(image_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, click_x, click_y, bbox, coco_json, created_at
        FROM annotations
        WHERE image_id = ?
        ORDER BY created_at DESC
    """, (image_id,))
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "click": [r[1], r[2]],
            "bbox": json.loads(r[3]),
            "coco": json.loads(r[4]),
            "created_at": r[5]
        }
        for r in rows
    ]
