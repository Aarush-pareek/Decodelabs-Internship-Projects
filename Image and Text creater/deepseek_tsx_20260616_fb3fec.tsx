"use client";
import { useEffect, useState } from "react";
import { getHistory } from "@/hooks/use-api";

export default function HistoryPage() {
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    getHistory().then(setHistory).catch(console.error);
  }, []);

  return (
    <div>
      <h2 className="text-3xl font-bold mb-8">Processing History</h2>
      <div className="glass rounded-2xl overflow-hidden">
        <table className="w-full text-left">
          <thead className="bg-white/5">
            <tr>
              <th className="p-4">Type</th>
              <th className="p-4">Image</th>
              <th className="p-4">Confidence</th>
              <th className="p-4">Date</th>
            </tr>
          </thead>
          <tbody>
            {history.map((h: any) => (
              <tr key={h.id} className="border-t border-white/5 hover:bg-white/5 transition">
                <td className="p-4 capitalize">{h.type.replace("_"," ")}</td>
                <td className="p-4">{h.image_name}</td>
                <td className="p-4">{h.confidence}%</td>
                <td className="p-4">{new Date(h.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}