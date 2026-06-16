import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import EnginePage from './pages/RecommendationEnginePage';
import AnalyticsPage from './pages/AnalyticsDashboardPage';
import AboutPage from './pages/AboutPage';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/engine" element={<EnginePage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </Layout>
  );
}

export default App;