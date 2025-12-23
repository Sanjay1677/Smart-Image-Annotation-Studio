import React, { useState } from "react";
import { uploadImage } from "./api";
import ImageCanvas from "./ImageCanvas";
import "./App.css";

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [imageURL, setImageURL] = useState(null);
  const [imageId, setImageId] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setImageURL(URL.createObjectURL(file));
    }
  };

  const handleUpload = async () => {
    if (!imageFile) return alert("Choose an image first!");

    setLoading(true);

    try {
      const res = await uploadImage(imageFile);
      setImageId(res.data.image_id);
    } catch (err) {
      console.error(err);
      alert("Upload failed");
    }

    setLoading(false);
  };

  return (
    <div className="app-container">
      <h1>Smart Image Annotation – SAM Frontend</h1>

      <input type="file" accept="image/*" onChange={handleFileSelect} />

      {imageURL && (
        <button className="upload-btn" onClick={handleUpload}>
          Upload Image
        </button>
      )}

      {loading && <div className="loader"></div>}

      {/* Once uploaded successfully, show segmentation canvas */}
      {imageId && imageURL && (
        <ImageCanvas imageSrc={imageURL} imageId={imageId} />
      )}
    </div>
  );
}

export default App;
