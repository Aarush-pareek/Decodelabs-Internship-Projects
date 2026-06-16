import { Search } from 'lucide-react';

const SearchBar = ({ value, onChange, placeholder = "Search recommendations..." }) => (
  <div className="relative w-full max-w-md">
    <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 w-5 h-5" />
    <input
      type="text"
      value={value}
      onChange={e => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-full pl-10 pr-4 py-3 glass rounded-full text-sm outline-none focus:ring-2 focus:ring-primary transition"
    />
  </div>
);

export default SearchBar;