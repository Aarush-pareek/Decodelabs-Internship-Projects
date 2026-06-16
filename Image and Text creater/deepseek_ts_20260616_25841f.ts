import type { Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        navy: "#0B1120",
        electric: "#6366F1",
        purple: "#8B5CF6",
      },
    },
  },
  plugins: [],
} satisfies Config;