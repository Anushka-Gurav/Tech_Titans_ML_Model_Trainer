import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Progress } from '../components/ui/progress';
import { Loader2, CheckCircle, XCircle } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

export default function TrainingProgress() {
  const navigate = useNavigate();
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('initializing');
  const [message, setMessage] = useState('Starting training...');
  const jobId = sessionStorage.getItem('jobId');

  useEffect(() => {
    if (!jobId) {
      navigate('/select-model');
      return;
    }

    const interval = setInterval(() => {
      checkProgress();
    }, 2000);

    return () => clearInterval(interval);
  }, [jobId]);

  const checkProgress = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/model/progress/${jobId}`);
      const data = response.data;
      
      setProgress(data.progress);
      setStatus(data.status);
      setMessage(data.message);

      if (data.status === 'completed') {
        toast.success('Training completed!');
        setTimeout(() => {
          const isComparison = sessionStorage.getItem('isComparison') === 'true';
          if (isComparison) {
            navigate(`/comparison-results/${jobId}`);
          } else {
            navigate(`/results/${jobId}`);
          }
        }, 1000);
      } else if (data.status === 'failed') {
        toast.error(`Training failed: ${data.message}`);
      }
    } catch (error) {
      console.error('Failed to check progress:', error);
    }
  };

  return (
    <div className="min-h-screen noise-bg flex items-center justify-center">
      <div className="max-w-3xl w-full mx-auto px-4">
        <div className="glass rounded-lg p-12" data-testid="training-progress-container">
          <div className="text-center mb-8">
            {status === 'running' || status === 'initializing' ? (
              <Loader2 className="h-16 w-16 text-primary mx-auto mb-4 animate-spin" data-testid="training-spinner" />
            ) : status === 'completed' ? (
              <CheckCircle className="h-16 w-16 text-success mx-auto mb-4" data-testid="training-success-icon" />
            ) : status === 'failed' ? (
              <XCircle className="h-16 w-16 text-error mx-auto mb-4" data-testid="training-error-icon" />
            ) : null}
            
            <h1 className="text-3xl sm:text-4xl font-heading font-bold mb-2" data-testid="training-status">
              {status === 'completed'
                ? 'Training Complete!'
                : status === 'failed'
                ? 'Training Failed'
                : 'Training in Progress'}
            </h1>
            <p className="text-textMuted" data-testid="training-message">{message}</p>
          </div>

          <div className="space-y-4">
            <Progress value={progress} className="h-4" data-testid="training-progress-bar" />
            <div className="text-center text-2xl font-heading font-bold text-primary" data-testid="training-progress-percent">
              {progress}%
            </div>
          </div>

          {/* Progress Steps */}
          <div className="mt-8 space-y-3">
            {[
              { step: 'Loading dataset', threshold: 10 },
              { step: 'Preparing data', threshold: 30 },
              { step: 'Training model', threshold: 50 },
              { step: 'Evaluating model', threshold: 70 },
              { step: 'Saving results', threshold: 90 },
              { step: 'Complete', threshold: 100 },
            ].map((item, idx) => (
              <div
                key={idx}
                className={`flex items-center space-x-3 text-sm ${
                  progress >= item.threshold ? 'text-primary' : 'text-textMuted'
                }`}
                data-testid={`training-step-${idx}`}
              >
                {progress >= item.threshold ? (
                  <CheckCircle className="h-5 w-5" />
                ) : progress >= item.threshold - 20 ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <div className="h-5 w-5 rounded-full border-2 border-current" />
                )}
                <span>{item.step}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}