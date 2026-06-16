import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  RadarController,
  RadialLinearScale,
  PointElement,
  LineElement,
} from 'chart.js';
import { Pie, Bar } from 'react-chartjs-2';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, RadarController, RadialLinearScale, PointElement, LineElement);

export const InterestPieChart = ({ data }) => {
  const chartData = {
    labels: Object.keys(data),
    datasets: [{
      data: Object.values(data),
      backgroundColor: ['#6366F1', '#8B5CF6', '#06B6D4', '#F59E0B', '#EF4444', '#10B981', '#EC4899'],
      borderWidth: 0,
    }],
  };
  return <Pie data={chartData} options={{ plugins: { legend: { position: 'bottom' } } }} />;
};

export const CategoryBarChart = ({ data }) => {
  const chartData = {
    labels: Object.keys(data),
    datasets: [{
      label: 'Items per Category',
      data: Object.values(data),
      backgroundColor: '#6366F1',
      borderRadius: 8,
    }],
  };
  return <Bar data={chartData} options={{ responsive: true }} />;
};