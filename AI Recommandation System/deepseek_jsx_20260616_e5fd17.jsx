import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Sparkles, ArrowRight, Cpu, BarChart3, Shield } from 'lucide-react';

const HomePage = () => (
  <div className="text-center">
    {/* Hero */}
    <motion.div
      initial={{ opacity: 0, y: -50 }}
      animate={{ opacity: 1, y: 0 }}
      className="py-20"
    >
      <h1 className="text-6xl font-extrabold bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent">
        AI Recommendation Logic System
      </h1>
      <p className="mt-6 text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
        Intelligent preference analysis, similarity scoring, and personalized recommendations – built for the modern AI era.
      </p>
      <Link to="/engine">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="mt-10 px-8 py-4 bg-primary text-white rounded-full font-semibold flex items-center gap-2 mx-auto shadow-lg shadow-primary/30"
        >
          Try Recommendation Engine <ArrowRight className="w-5 h-5" />
        </motion.button>
      </Link>
    </motion.div>

    {/* Features */}
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-20">
      {[
        { icon: <Cpu className="w-8 h-8 text-primary" />, title: 'Pattern Matching', desc: 'Advanced similarity scoring using weighted tag and category matching.' },
        { icon: <BarChart3 className="w-8 h-8 text-secondary" />, title: 'Real-time Analytics', desc: 'Visual dashboards showing interest distribution and recommendation confidence.' },
        { icon: <Shield className="w-8 h-8 text-accent" />, title: 'Explainable AI', desc: 'Every recommendation comes with a reason and match percentage.' },
      ].map((feat, idx) => (
        <motion.div
          key={idx}
          whileHover={{ y: -10 }}
          className="glass rounded-2xl p-8 text-left"
        >
          <div className="mb-4">{feat.icon}</div>
          <h3 className="text-xl font-bold">{feat.title}</h3>
          <p className="mt-2 text-gray-500 dark:text-gray-400">{feat.desc}</p>
        </motion.div>
      ))}
    </div>
  </div>
);

export default HomePage;