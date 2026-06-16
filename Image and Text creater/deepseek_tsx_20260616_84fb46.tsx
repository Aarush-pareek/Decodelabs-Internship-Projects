"use client";
import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { uploadOCR } from "@/hooks/use-api";
import { Upload, FileText, AlertTriangle, Download } from "lucide-react";

export default function OCRPage() {
  const [image, setImage] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    setImage(URL.createObjectURL(file));
    setLoading(true);
    try {
      const data = await uploadOCR(file);
      setResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: {"image/*": []} });

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-3xl font-bold mb-6 flex items-center gap-2">
        <FileText className="text-electric" /> OCR Studio
      </h2>
      <div {...getRootProps()} className="glass border-dashed border-2 border-white/20 rounded-2xl p-10 text-center cursor-pointer hover:border-electric transition">
        <input {...getInputProps()} />
        <Upload className="mx-auto mb-4 text-gray-400" size={40} />
        {isDragActive ? <p>Drop the image here...</p> : <p>Drag & drop an image, or click to select</p>}
      </div>
      {image && <img src={image} alt="preview" className="mt-6 rounded-xl max-h-80 mx-auto" />}
      {loading && <div className="mt-6 text-center text-electric animate-pulse">Processing...</div>}
      {result && (
        <div className="mt-8 glass p-6 rounded-2xl">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-semibold">Extracted Text</h3>
            {result.warning && (
              <span className="flex items-center gap-1 text-yellow-500 text-sm"><AlertTriangle size={16}/> {result.warning}</span>
            )}
          </div>
          <pre className="whitespace-pre-wrap text-gray-300 bg-black/20 p-4 rounded-lg">{result.text || "No text found"}</pre>
          <div className="mt-4 flex justify-between items-center">
            <p>Confidence: <span className="text-electric font-bold">{result.confidence}%</span></p>
            <button onClick={() => {
              const blob = new Blob([result.text], {type: "text/plain"});
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url; a.download = "ocr-result.txt"; a.click();
            }} className="flex items-center gap-2 bg-electric px-4 py-2 rounded-full text-sm">
              <Download size={16} /> Export
            </button>
          </div>
        </div>
      )}
    </div>
  );
}