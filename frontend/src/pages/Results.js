import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Button } from '../components/ui/button';
import { Download, Home, RefreshCw } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

export default function Results() {
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
        navigate('/training');
      }
    } catch (error) {
      toast.error('Failed to load results');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    window.open(`${BACKEND_URL}/api/model/download/${jobId}`, '_blank');
    toast.success('Model download started');
  };

  if (loading) {
    return (
      <div className="min-h-screen noise-bg flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin h-12 w-12 border-4 border-primary border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-textMuted">Loading results...</p>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="min-h-screen noise-bg flex items-center justify-center">
        <div className="text-center">
          <p className="text-error text-xl">Results not found</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen noise-bg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-4xl sm:text-5xl font-heading font-bold mb-2" data-testid="results-title">
              Training <span className="text-primary">Results</span>
            </h1>
            <p className="text-textMuted">Job ID: {jobId}</p>
          </div>
          <div className="flex gap-4">
            <Button
              data-testid="download-model-btn"
              onClick={handleDownload}
              className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-glow"
            >
              <Download className="mr-2 h-4 w-4" />
              Download Model (.pkl)
            </Button>
            <Button
              data-testid="train-another-btn"
              onClick={() => {
                sessionStorage.clear();
                navigate('/select-model');
              }}
              variant="outline"
              className="border-border hover:bg-surface"
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Train Another
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

        {/* Metrics */}
        <div className="glass rounded-lg p-8 mb-8" data-testid="metrics-container">
          <h2 className="text-2xl font-heading font-bold mb-6">Performance Metrics</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {Object.entries(results.metrics).map(([key, value]) => (
              <div
                key={key}
                className="bg-surface/50 rounded-lg p-6 border border-border/50"
                data-testid={`metric-${key}`}
              >
                <div className="text-sm text-textMuted uppercase mb-2">
                  {key.replace(/_/g, ' ')}
                </div>
                <div className="text-3xl font-heading font-bold text-primary">
                  {typeof value === 'number' ? value.toFixed(4) : value}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Visualizations */}
        {results.visualizations && Object.keys(results.visualizations).length > 0 && (
          <div className="glass rounded-lg p-8" data-testid="visualizations-container">
            <h2 className="text-2xl font-heading font-bold mb-6">Visualizations</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {Object.entries(results.visualizations).map(([key, imgData]) => (
                <div key={key} className="bg-surface/50 rounded-lg p-4" data-testid={`viz-${key}`}>
                  <h3 className="text-lg font-heading font-bold mb-4 capitalize">
                    {key.replace(/_/g, ' ')}
                  </h3>
                  <img
                    src={`data:image/png;base64,${imgData}`}
                    alt={key}
                    className="w-full rounded border border-border/50"
                  />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}