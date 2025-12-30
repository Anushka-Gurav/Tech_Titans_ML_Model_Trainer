import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Button } from '../components/ui/button';
import { Home, RefreshCw, Trophy, TrendingUp } from 'lucide-react';
import { toast } from 'sonner';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

export default function ComparisonResults() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [results, setResults] = useState(null);

  useEffect(() => {
    fetchResults();
  }, [jobId]);

  const fetchResults = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/model/progress/${jobId}`);
      if (response.data.status === 'completed' && response.data.result) {
        setResults(response.data.result);
      } else {
        toast.error('Results not ready yet');
      }
    } catch (error) {
      toast.error('Failed to load results');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen noise-bg flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin h-12 w-12 border-4 border-primary border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-textMuted">Loading comparison results...</p>
        </div>
      </div>
    );
  }

  if (!results || !results.comparison) {
    return (
      <div className="min-h-screen noise-bg flex items-center justify-center">
        <div className="text-center">
          <p className="text-error text-xl">Comparison results not found</p>
        </div>
      </div>
    );
  }

  const { model1_name, model2_name, model1_metrics, model2_metrics, winner } = results.comparison;
  
  // Prepare chart data
  const chartData = Object.keys(model1_metrics).map(key => ({
    metric: key.replace(/_/g, ' ').toUpperCase(),
    [model1_name]: model1_metrics[key],
    [model2_name]: model2_metrics[key]
  }));

  return (
    <div className="min-h-screen noise-bg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-4xl sm:text-5xl font-heading font-bold mb-2" data-testid="results-title">
              Model <span className="text-primary">Comparison</span>
            </h1>
            <p className="text-textMuted">Comparing {model1_name} vs {model2_name}</p>
          </div>
          <div className="flex gap-4">
            <Button
              data-testid="compare-again-btn"
              onClick={() => {
                sessionStorage.clear();
                navigate('/compare-models');
              }}
              variant="outline"
              className="border-border hover:bg-surface"
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Compare Again
            </Button>
            <Button
              data-testid="back-home-btn"
              onClick={() => navigate('/')}
              variant="outline"
              className="border-border hover:bg-surface"
            >
              <Home className="mr-2 h-4 w-4" />
              Home
            </Button>
          </div>
        </div>

        {/* Winner Banner */}
        <div className="glass rounded-lg p-6 mb-8 border-2 border-primary" data-testid="winner-banner">
          <div className="flex items-center gap-4">
            <Trophy className="h-12 w-12 text-accent" />
            <div>
              <h2 className="text-2xl font-heading font-bold mb-1">
                Winner: <span className="text-primary">{winner}</span>
              </h2>
              <p className="text-textMuted">Based on overall performance metrics</p>
            </div>
          </div>
        </div>

        {/* Side-by-side Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          <div className="glass rounded-lg p-8" data-testid="model1-metrics">
            <div className="flex items-center gap-3 mb-6">
              <div className="h-3 w-3 rounded-full bg-primary" />
              <h2 className="text-2xl font-heading font-bold">{model1_name}</h2>
            </div>
            <div className="space-y-4">
              {Object.entries(model1_metrics).map(([key, value]) => (
                <div key={key} className="flex justify-between items-center">
                  <span className="text-textMuted capitalize">{key.replace(/_/g, ' ')}</span>
                  <span className="text-2xl font-heading font-bold text-primary">
                    {typeof value === 'number' ? value.toFixed(4) : value}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="glass rounded-lg p-8" data-testid="model2-metrics">
            <div className="flex items-center gap-3 mb-6">
              <div className="h-3 w-3 rounded-full bg-secondary" />
              <h2 className="text-2xl font-heading font-bold">{model2_name}</h2>
            </div>
            <div className="space-y-4">
              {Object.entries(model2_metrics).map(([key, value]) => (
                <div key={key} className="flex justify-between items-center">
                  <span className="text-textMuted capitalize">{key.replace(/_/g, ' ')}</span>
                  <span className="text-2xl font-heading font-bold text-secondary">
                    {typeof value === 'number' ? value.toFixed(4) : value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Comparison Chart */}
        <div className="glass rounded-lg p-8" data-testid="comparison-chart">
          <h2 className="text-2xl font-heading font-bold mb-6 flex items-center gap-2">
            <TrendingUp className="h-6 w-6 text-primary" />
            Performance Comparison
          </h2>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="metric" stroke="#a1a1aa" />
              <YAxis stroke="#a1a1aa" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#18181b',
                  border: '1px solid #27272a',
                  borderRadius: '0.5rem',
                  color: '#f4f4f5'
                }}
              />
              <Legend />
              <Bar dataKey={model1_name} fill="#4ade80" />
              <Bar dataKey={model2_name} fill="#a78bfa" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}