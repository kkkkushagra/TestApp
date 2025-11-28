// src/api.js
import axios from "axios";

const API_BASE = "http://127.0.0.1:8000"; // FastAPI backend

export const analyzeContract = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await axios.post(`${API_BASE}/redline/analyze/`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data;
  } catch (err) {
    console.error("Error analyzing contract:", err);
    throw err.response?.data || { detail: "Server error" };
  }
};
