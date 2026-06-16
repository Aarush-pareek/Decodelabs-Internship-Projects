import Navbar from './Navbar';
import { motion } from 'framer-motion';

const Layout = ({ children }) => (
  <div className="min-h-screen bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-darkbg dark:to-slate-900 transition-colors duration-500">
    <Navbar />
    <motion.main
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="container mx-auto px-4 py-8"
    >
      {children}
    </motion.main>
  </div>
);

export default Layout;