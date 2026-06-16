const InterestChip = ({ label, selected, onClick }) => (
  <button
    onClick={onClick}
    className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 glass hover:scale-105 ${
      selected
        ? 'bg-primary text-white shadow-lg shadow-primary/30 border-primary'
        : 'hover:bg-primary/10 border-gray-300 dark:border-gray-600'
    }`}
  >
    {label}
  </button>
);

export default InterestChip;