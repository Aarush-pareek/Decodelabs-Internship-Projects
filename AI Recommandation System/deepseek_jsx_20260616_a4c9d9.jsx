import { Link, useLocation } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import { Sun, Moon, Sparkles } from 'lucide-react';

const Navbar = () => {
  const { darkMode, toggleTheme } = useTheme();
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Home' },
    { path: '/engine', label: 'Engine' },
    { path: '/analytics', label: 'Analytics' },
    { path: '/about', label: 'About' },
  ];

  return (
    <nav className="sticky top-0 z-50 glass backdrop-blur-xl border-b border-white/10 dark:border-white/5">
      <div className="container mx-auto flex items-center justify-between px-4 py-3">
        <Link to="/" className="flex items-center gap-2 text-2xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
          <Sparkles className="w-6 h-6 text-primary" />
          AI Recommend
        </Link>
        <div className="flex items-center gap-6">
          {navItems.map(item => (
            <Link
              key={item.path}
              to={item.path}
              className={`text-sm font-medium transition-colors ${
                location.pathname === item.path
                  ? 'text-primary'
                  : 'text-gray-600 dark:text-gray-300 hover:text-primary'
              }`}
            >
              {item.label}
            </Link>
          ))}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-full glass hover:bg-white/20 transition"
          >
            {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;