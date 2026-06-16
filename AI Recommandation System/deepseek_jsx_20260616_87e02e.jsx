import { motion } from 'framer-motion';

const StatCard = ({ title, value, icon, color }) => (
  <motion.div
    whileHover={{ scale: 1.05 }}
    className="glass rounded-xl p-5 flex items-center gap-4"
  >
    <div className={`p-3 rounded-full ${color} bg-opacity-20`}>
      {icon}
    </div>
    <div>
      <p className="text-sm text-gray-500 dark:text-gray-400">{title}</p>
      <p className="text-2xl font-bold">{value}</p>
    </div>
  </motion.div>
);

export default StatCard;