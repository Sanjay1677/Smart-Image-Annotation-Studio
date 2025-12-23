import React, { useRef, useState, useEffect } from "react";
import { segmentPoint, segmentBox } from "./api";

export default function ImageCanvas({ imageSrc, imageId }) {
  const canvasRef = useRef();
  const imgRef = useRef(new Image());

  const [isDragging, setDragging] = useState(false);
  const [start, setStart] = useState(null);
  const [scale, setScale] = useState(1);
  const [croppedMask, setCroppedMask] = useState(null);
  const [fileName, setFileName] = useState("masked_output.png");
  const [segLoading, setSegLoading] = useState(false);
  const [cocoData, setCocoData] = useState(null);


  useEffect(() => {
    if (!imageSrc) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = imgRef.current;

    img.onload = () => {
      const maxW = 900;
      const maxH = 650;
      const s = Math.min(maxW / img.width, maxH / img.height);

      canvas.width = img.width * s;
      canvas.height = img.height * s;
      setScale(s);

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };

    img.src = imageSrc;
  }, [imageSrc]);

  const getCoords = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: Math.round((e.clientX - rect.left) / scale),
      y: Math.round((e.clientY - rect.top) / scale),
    };
  };

  const handleMouseDown = (e) => {
    setStart(getCoords(e));
    setDragging(true);
  };

  const handleMouseMove = (e) => {
    if (!isDragging || !start) return;

    const ctx = canvasRef.current.getContext("2d");
    ctx.drawImage(imgRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);

    const pos = getCoords(e);
    ctx.strokeStyle = "cyan";
    ctx.lineWidth = 2;
    ctx.strokeRect(
      start.x * scale,
      start.y * scale,
      (pos.x - start.x) * scale,
      (pos.y - start.y) * scale
    );
  };

  const handleMouseUp = async (e) => {
    if (!imageId) return;

    setDragging(false);
    const end = getCoords(e);
    setSegLoading(true);

    try {
      let res;
      if (Math.abs(end.x - start.x) < 5 && Math.abs(end.y - start.y) < 5) {
        res = await segmentPoint(imageId, start.x, start.y);
      } else {
        res = await segmentBox(imageId, {
          x1: start.x,
          y1: start.y,
          x2: end.x,
          y2: end.y,
        });
      }

      setCroppedMask(`data:image/png;base64,${res.data.raw_mask_base64}`);
      setFileName(res.data.filename || "masked_output.png");
      setCocoData(res.data.coco);

    } catch (err) {
      alert("Segmentation failed");
    }

    setSegLoading(false);
  };

  const downloadMask = () => {
    const link = document.createElement("a");
    link.href = croppedMask;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const downloadCOCO = () => {
    const blob = new Blob(
      [JSON.stringify(cocoData, null, 2)],
      { type: "application/json" }
    );
  
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = fileName.replace(".png", ".json");
  
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  

  return (
    <div className="canvas-wrap">
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        className="canvas-box"
      />

      <div className="output-box">
        <h3>Masked Output</h3>

        {segLoading && <div>Processing...</div>}

        {croppedMask && (
  <>
    <img src={croppedMask} alt="Masked" className="mask-image" />
    <div
      style={{
        display: "flex",
        gap: 10,
        marginTop: 10,
      }}
    >
      <button
        onClick={downloadMask}
        style={{
          padding: "8px 14px",
          background: "#0af",
          border: "none",
          borderRadius: 6,
          fontWeight: "bold",
          cursor: "pointer",
        }}
      >
        ⬇ Download Mask
      </button>
      <button
        onClick={downloadCOCO}
        style={{
          padding: "8px 14px",
          background: "#0af",
          border: "none",
          borderRadius: 6,
          fontWeight: "bold",
          cursor: "pointer",
        }}
      >
        ⬇ Download COCO
      </button>
    </div>
  </>
)}
      </div>
    </div>
  );
}
