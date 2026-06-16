import Link from "next/link";
import { FileText, ScanEye, BarChart3, History, Home } from "lucide-react";

const sidebarItems = [
  { href: "/dashboard/ocr", label: "OCR Studio", icon: <FileText size={20} /> },
  { href: "/dashboard/object-detection", label: "Object Detection", icon: <ScanEye size={20} /> },
  { href: "/dashboard/analytics", label: "Analytics", icon: <BarChart3 size={20} /> },
  { href: "/dashboard/history", label: "History", icon: <History size={20} /> },
  { href: "/", label: "Home", icon: <Home size={20} /> },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex bg-navy">
      <aside className="w-64 glass border-r border-white/10 hidden md:block">
        <div className="p-6 text-2xl font-bold bg-gradient-to-r from-electric to-purple bg-clip-text text-transparent">
          AI Vision
        </div>
        <nav className="mt-10 space-y-2 px-4">
          {sidebarItems.map(item => (
            <Link key={item.href} href={item.href}
              className="flex items-center gap-3 p-3 rounded-lg hover:bg-white/5 text-gray-300 hover:text-white transition">
              {item.icon}
              <span>{item.label}</span>
            </Link>
          ))}
        </nav>
      </aside>
      <main className="flex-1 p-6 md:p-10 overflow-auto">{children}</main>
    </div>
  );
}