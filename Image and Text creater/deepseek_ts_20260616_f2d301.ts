import { API_URL } from "@/lib/utils";

export async function uploadOCR(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_URL}/api/ocr`, { method: "POST", body: formData });
  return res.json();
}

export async function uploadDetection(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_URL}/api/object-detection`, { method: "POST", body: formData });
  return res.json();
}

export async function getHistory() {
  const res = await fetch(`${API_URL}/api/history`);
  return res.json();
}

export async function getAnalytics() {
  const res = await fetch(`${API_URL}/api/analytics`);
  return res.json();
}