// src/api.js
const API_URL = "http://127.0.0.1:5000";

export async function submitDaily(text, audioFile) {
  const formData = new FormData();
  formData.append("text", text);

  if (audioFile) {
    formData.append("audio", audioFile);
  }

  const response = await fetch(`${API_URL}/submit-daily`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    throw new Error("Backend not reachable");
  }

  return response.json();
}

