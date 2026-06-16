"use client";
import { useEffect, useState } from "react";
import { getAnalytics } from "@/hooks/use-api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";

export default function AnalyticsPage() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    getAnalytics().then(setData).catch(console.error);
  }, []);

  if (!data) return <div>Loading analytics...</div>;

  const chartData = [
    { name: "OCR Requests", value: data.total_ocr, fill: "#6366F1" },
    { name: "Obj Detection Requests", value: data.total_object_detection, fill: "#8B5CF6" },
  ];

  return (
    <div>
      <h2 className="text-3xl font-bold mb-8">Analytics Dashboard</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
        {[
          { label: "Total OCR", value: data.total_ocr },
          { label: "Total Object Detections", value: data.total_object_detection },
          { label: "OCR Avg Confidence", value: `${data.ocr_avg_confidence}%` },
        ].map((s, i) => (
          <div key={i} className="glass p-6 rounded-2xl">
            <p className="text-gray-400">{s.label}</p>
            <p className="text-3xl font-bold text-white">{s.value}</p>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="glass p-6 rounded-2xl">
          <h3 className="text-lg font-semibold mb-4">Requests Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={chartData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label>
                {chartData.map((entry, index) => (
                  <Cell key={index} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="glass p-6 rounded-2xl">
          <h3 className="text-lg font-semibold mb-4">Confidence Metrics</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={[
              { name: "OCR", confidence: data.ocr_avg_confidence },
              { name: "Object Detection", confidence: data.obj_avg_confidence },
            ]}>
              <XAxis dataKey="name" stroke="#fff" />
              <YAxis stroke="#fff" />
              <Tooltip />
              <Bar dataKey="confidence" fill="#6366F1" radius={[8,8,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}