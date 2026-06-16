/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: '#6366F1',
        secondary: '#8B5CF6',
        accent: '#06B6D4',
        darkbg: '#0F172A',
        lighttext: '#F8FAFC',
      },
      backdropBlur: {
        xs: '2px',
      }
    },
  },
  plugins: [],
}