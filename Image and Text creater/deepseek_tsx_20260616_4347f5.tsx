"use client";
import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { uploadDetection } from "@/hooks/use-api";
import { ScanEye, AlertTriangle } from "lucide-react";

export default function ObjectDetectionPage() {
  const [image, setImage] = useState<string | null>(null);
  const [annotated, setAnnotated] = useState<string | null>(null);
  const [detections, setDetections] = useState<any[]>([]);
  const [avgConf, setAvgConf] = useState(0);
  const [warning, setWarning] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    setImage(URL.createObjectURL(file));
    setLoading(true);
    try {
      const data = await uploadDetection(file);
      setAnnotated(`data:image/jpeg;base64,${data.annotated_image}`);
      setDetections(data.detections);
      setAvgConf(data.average_confidence);
      setWarning(data.warning);
    } finally {
      setLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: {"image/*": []} });

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-3xl font-bold mb-6 flex items-center gap-2">
        <ScanEye className="text-purple" /> Object Detection
      </h2>
      <div {...getRootProps()} className="glass border-dashed border-2 border-white/20 rounded-2xl p-10 text-center cursor-pointer hover:border-purple transition">
        <input {...getInputProps()} />
        <ScanEye className="mx-auto mb-4 text-gray-400" size={40} />
        {isDragActive ? <p>Drop the image here...</p> : <p>Drag & drop an image, or click to select</p>}
      </div>
      {image && <img src={image} alt="original" className="mt-6 rounded-xl max-h-80 mx-auto" />}
      {loading && <div className="mt-6 text-center text-purple animate-pulse">Analyzing...</div>}
      {annotated && (
        <div className="mt-8 glass p-6 rounded-2xl">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-semibold">Detection Results</h3>
            {warning && <span className="flex items-center gap-1 text-yellow-500"><AlertTriangle size={16}/> {warning}</span>}
          </div>
          <img src={annotated} alt="detected" className="rounded-xl mb-4" />
          <p>Average Confidence: <span className="text-purple font-bold">{avgConf}%</span></p>
          <ul className="mt-3 space-y-2">
            {detections.map((d, i) => (
              <li key={i} className="flex justify-between bg-black/20 p-2 rounded-lg">
                <span>{d.label}</span>
                <span className="text-electric">{d.confidence}%</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}