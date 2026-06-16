import axios from 'axios';

const API_BASE = 'http://localhost:5000/api';

export const fetchItems = () => axios.get(`${API_BASE}/items`);

export const getRecommendations = (interests) =>
  axios.post(`${API_BASE}/recommend`, { interests });

export const getAnalytics = (interests) =>
  axios.post(`${API_BASE}/analytics`, { interests });