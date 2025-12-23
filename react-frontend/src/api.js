import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

export const uploadImage = (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return API.post("/upload", formData);
};

export const segmentPoint = (imageId, x, y) =>
  API.post("/segment/point", { image_id: imageId, x, y });

export const segmentBox = (imageId, coords) =>
  API.post("/segment/box", {
    image_id: imageId,
    ...coords,
  });
