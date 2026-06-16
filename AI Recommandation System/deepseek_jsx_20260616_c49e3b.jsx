import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { getRecommendations, fetchItems } from '../services/api';
import InterestChip from '../components/InterestChip';
import RecommendationCard from '../components/RecommendationCard';
import SearchBar from '../components/SearchBar';
import FilterSortControls from '../components/FilterSortControls';
import { Sparkles, RefreshCw } from 'lucide-react';

const interestsList = [
  "Technology", "AI", "Gaming", "Fitness", "Music", "Movies",
  "Business", "Finance", "Sports", "Programming", "Design",
  "Data Science", "Robotics", "Writing", "Cooking", "Travel",
  "Photography", "Education", "Health", "Science"
];

const EnginePage = () => {
  const [selectedInterests, setSelectedInterests] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [allCategories, setAllCategories] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('');
  const [sortBy, setSortBy] = useState('score');
  const [loading, setLoading] = useState(false);

  // Fetch all items once to extract categories
  useEffect(() => {
    fetchItems().then(res => {
      const cats = [...new Set(res.data.map(item => item.category))];
      setAllCategories(cats);
    }).catch(() => {});
  }, []);

  const toggleInterest = (interest) => {
    setSelectedInterests(prev =>
      prev.includes(interest) ? prev.filter(i => i !== interest) : [...prev, interest]
    );
  };

  // Dynamic recommendations when interests change (instant update)
  useEffect(() => {
    if (selectedInterests.length === 0) {
      setRecommendations([]);
      return;
    }
    setLoading(true);
    getRecommendations(selectedInterests)
      .then(res => {
        setRecommendations(res.data);
      })
      .catch(() => setRecommendations([]))
      .finally(() => setLoading(false));
  }, [selectedInterests]);

  // Filter and sort displayed recommendations
  let filtered = recommendations.filter(r =>
    r.name.toLowerCase().includes(searchTerm.toLowerCase()) &&
    (categoryFilter === '' || r.category === categoryFilter)
  );

  if (sortBy === 'rating') filtered.sort((a, b) => b.rating - a.rating);
  else if (sortBy === 'name') filtered.sort((a, b) => a.name.localeCompare(b.name));
  // default score sort already from API

  return (
    <div className="max-w-7xl mx-auto">
      <h2 className="text-3xl font-bold mb-8 flex items-center gap-2">
        <Sparkles className="text-primary" /> Recommendation Engine
      </h2>

      {/* Interest Selection */}
      <div className="glass rounded-2xl p-6 mb-8">
        <h3 className="text-lg font-semibold mb-4">Select Your Interests</h3>
        <div className="flex flex-wrap gap-3">
          {interestsList.map(interest => (
            <InterestChip
              key={interest}
              label={interest}
              selected={selectedInterests.includes(interest)}
              onClick={() => toggleInterest(interest)}
            />
          ))}
        </div>
        {selectedInterests.length > 0 && (
          <p className="mt-3 text-sm text-gray-500">
            {selectedInterests.length} interest{selectedInterests.length>1?'s':''} selected
          </p>
        )}
      </div>

      {/* Results Area */}
      {selectedInterests.length > 0 ? (
        <>
          <div className="flex flex-wrap gap-4 items-center justify-between mb-6">
            <SearchBar value={searchTerm} onChange={setSearchTerm} />
            <FilterSortControls
              categories={allCategories}
              selectedCategory={categoryFilter}
              onCategoryChange={setCategoryFilter}
              sortBy={sortBy}
              onSortChange={setSortBy}
            />
          </div>

          {loading ? (
            <div className="flex justify-center py-20">
              <RefreshCw className="animate-spin text-primary w-12 h-12" />
            </div>
          ) : (
            <motion.div layout className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
              <AnimatePresence>
                {filtered.map((item, idx) => (
                  <RecommendationCard key={item.name} item={item} index={idx} />
                ))}
              </AnimatePresence>
            </motion.div>
          )}

          {filtered.length === 0 && !loading && (
            <p className="text-center text-gray-500 mt-20">No recommendations match your filters.</p>
          )}
        </>
      ) : (
        <div className="text-center py-20 text-gray-500">
          Select at least one interest to see recommendations.
        </div>
      )}
    </div>
  );
};

export default EnginePage;