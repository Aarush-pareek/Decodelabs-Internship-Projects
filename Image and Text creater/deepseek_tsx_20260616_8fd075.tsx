import Link from "next/link";
import { ArrowRight, Sparkles, ScanEye, FileText, BarChart3 } from "lucide-react";

export default function LandingPage() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-4 text-center bg-gradient-to-br from-navy via-purple/20 to-navy">
      <nav className="absolute top-0 w-full flex justify-between p-6">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-electric to-purple bg-clip-text text-transparent">
          AI Vision Suite
        </h1>
        <div className="flex gap-4">
          <Link href="/dashboard/ocr" className="text-sm hover:text-electric transition">Dashboard</Link>
        </div>
      </nav>

      <div className="mt-20 max-w-4xl">
        <h2 className="text-5xl md:text-7xl font-extrabold leading-tight">
          Transform Images Into <span className="text-electric">Intelligence</span>
        </h2>
        <p className="mt-6 text-lg text-gray-400 max-w-2xl mx-auto">
          AI-powered OCR and Object Detection platform for modern businesses.
          Extract text from documents, detect objects in images, and gain insights with confidence analytics.
        </p>
        <div className="mt-10 flex flex-wrap gap-4 justify-center">
          <Link href="/dashboard/ocr" className="bg-electric hover:bg-electric/90 text-white px-8 py-4 rounded-full font-semibold inline-flex items-center gap-2 transition">
            Try Demo <ArrowRight size={18} />
          </Link>
          <Link href="/docs" className="glass px-8 py-4 rounded-full font-semibold inline-flex items-center gap-2 transition hover:bg-white/10">
            View Documentation
          </Link>
        </div>
      </div>

      {/* Features */}
      <div className="mt-32 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl w-full px-4">
        {[
          { icon: <FileText className="text-electric" size={32} />, title: "OCR Recognition", desc: "Extract text from invoices, receipts, books with high accuracy." },
          { icon: <ScanEye className="text-purple" size={32} />, title: "Object Detection", desc: "Detect persons, cars, bottles, chairs and more with bounding boxes." },
          { icon: <BarChart3 className="text-electric" size={32} />, title: "Confidence Analytics", desc: "Analyze AI confidence scores and track history." },
        ].map((f, i) => (
          <div key={i} className="glass p-8 rounded-2xl hover:scale-105 transition-transform duration-300">
            <div className="mb-4">{f.icon}</div>
            <h3 className="text-xl font-semibold">{f.title}</h3>
            <p className="mt-2 text-gray-400">{f.desc}</p>
          </div>
        ))}
      </div>
    </main>
  );
}