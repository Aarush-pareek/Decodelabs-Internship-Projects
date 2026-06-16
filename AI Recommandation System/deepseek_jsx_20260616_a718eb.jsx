import { useState, useEffect } from 'react';
import { getAnalytics } from '../services/api';
import { InterestPieChart, CategoryBarChart } from '../components/ChartComponents';
import StatCard from '../components/StatCard';
import { BarChart3, Target, Zap, TrendingUp } from 'lucide-react';

const sampleInterests = ["AI", "Programming", "Technology", "Business"];

const AnalyticsPage = () => {
  const [analytics, setAnalytics] = useState(null);
  const [interestsInput, setInterestsInput] = useState(sampleInterests.join(', '));
  const [loading, setLoading] = useState(false);

  const fetchAnalytics = (interestsArray) => {
    if (interestsArray.length === 0) return;
    setLoading(true);
    getAnalytics(interestsArray)
      .then(res => setAnalytics(res.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchAnalytics(sampleInterests);
  }, []);

  const handleUpdate = () => {
    const arr = interestsInput.split(',').map(s => s.trim()).filter(Boolean);
    fetchAnalytics(arr);
  };

  if (!analytics) return <div className="text-center py-20">Loading analytics...</div>;

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between mb-8">
        <h2 className="text-3xl font-bold flex items-center gap-2">
          <BarChart3 className="text-primary" /> Analytics Dashboard
        </h2>
        <div className="flex gap-2">
          <input
            value={interestsInput}
            onChange={e => setInterestsInput(e.target.value)}
            placeholder="e.g., AI, Programming"
            className="glass rounded-full px-4 py-2 text-sm outline-none"
          />
          <button onClick={handleUpdate} className="bg-primary text-white px-4 py-2 rounded-full text-sm">
            Update
          </button>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        <StatCard title="Total Interests" value={analytics.total_interests} icon={<Target className="text-white" />} color="bg-primary" />
        <StatCard title="Recommendations" value={analytics.recommendation_count} icon={<Zap className="text-white" />} color="bg-secondary" />
        <StatCard title="Avg Match Score" value={`${analytics.average_match_score}%`} icon={<TrendingUp className="text-white" />} color="bg-accent" />
        <StatCard title="Highest Match" value={analytics.highest_match?.name || '-'} icon={<BarChart3 className="text-white" />} color="bg-yellow-500" />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="glass rounded-2xl p-6">
          <h3 className="text-xl font-semibold mb-4">Interest Distribution</h3>
          <InterestPieChart data={analytics.interest_distribution} />
        </div>
        <div className="glass rounded-2xl p-6">
          <h3 className="text-xl font-semibold mb-4">Category Popularity</h3>
          <CategoryBarChart data={analytics.category_distribution} />
        </div>
      </div>

      {/* Recommendation Score Histogram can be added similarly */}
    </div>
  );
};

export default AnalyticsPage;