import { motion } from 'framer-motion';
import { Star, ThumbsUp, Zap } from 'lucide-react';

const confidenceColors = {
  'High Confidence': 'text-green-400',
  'Medium Confidence': 'text-yellow-400',
  'Low Confidence': 'text-red-400',
};

const RecommendationCard = ({ item, index }) => (
  <motion.div
    initial={{ opacity: 0, y: 30 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: index * 0.1 }}
    whileHover={{ scale: 1.02, y: -5 }}
    className="glass rounded-2xl p-6 flex flex-col justify-between hover:shadow-2xl hover:shadow-primary/10 transition-all"
  >
    <div>
      <div className="flex items-start justify-between">
        <h3 className="text-xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
          {item.name}
        </h3>
        <span className={`flex items-center gap-1 text-sm font-semibold ${confidenceColors[item.confidence]}`}>
          <Zap className="w-4 h-4" /> {item.confidence}
        </span>
      </div>
      <p className="text-sm text-gray-500 dark:text-gray-400 mt-1 capitalize">{item.category}</p>
      <div className="flex flex-wrap gap-2 mt-3">
        {item.tags.map(tag => (
          <span key={tag} className="px-2 py-0.5 bg-primary/10 text-primary text-xs rounded-full">
            {tag}
          </span>
        ))}
      </div>
      <div className="mt-4 text-sm italic text-gray-600 dark:text-gray-300">
        <ThumbsUp className="inline w-4 h-4 mr-1" />
        {item.reason}
      </div>
    </div>
    <div className="mt-4 flex items-center justify-between">
      <div className="flex items-center gap-1">
        <Star className="w-5 h-5 fill-yellow-400 text-yellow-400" />
        <span className="font-semibold">{item.rating}</span>
      </div>
      <div className="text-right">
        <div className="text-2xl font-bold text-primary">{item.score}%</div>
        <div className="text-xs text-gray-500">match</div>
      </div>
    </div>
    {/* Progress bar */}
    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-2">
      <div
        className="bg-gradient-to-r from-primary to-accent h-2 rounded-full transition-all duration-700"
        style={{ width: `${item.score}%` }}
      />
    </div>
  </motion.div>
);

export default RecommendationCard;