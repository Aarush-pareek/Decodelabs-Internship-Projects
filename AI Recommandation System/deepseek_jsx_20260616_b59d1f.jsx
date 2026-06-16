const FilterSortControls = ({ categories, selectedCategory, onCategoryChange, sortBy, onSortChange }) => (
  <div className="flex flex-wrap gap-4 items-center">
    <select
      value={selectedCategory}
      onChange={e => onCategoryChange(e.target.value)}
      className="glass rounded-full px-4 py-2 text-sm outline-none"
    >
      <option value="">All Categories</option>
      {categories.map(cat => (
        <option key={cat} value={cat}>{cat}</option>
      ))}
    </select>
    <select
      value={sortBy}
      onChange={e => onSortChange(e.target.value)}
      className="glass rounded-full px-4 py-2 text-sm outline-none"
    >
      <option value="score">Sort by Match Score</option>
      <option value="rating">Sort by Rating</option>
      <option value="name">Sort by Name</option>
    </select>
  </div>
);

export default FilterSortControls;