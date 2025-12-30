import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Button } from '../components/ui/button';
import { ArrowLeft, ArrowRight, Sparkles, Layers } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

export default function ModelSelection() {
  const navigate = useNavigate();
  const [modelType, setModelType] = useState(null); // 'supervised' or 'unsupervised'
  const [modelCategory, setModelCategory] = useState(null); // 'classification' or 'regression' for supervised
  const [selectedModel, setSelectedModel] = useState(null);
  const [models, setModels] = useState(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/models/list`);
      setModels(response.data);
    } catch (error) {
      toast.error('Failed to load models');
    }
  };

  const handleNext = () => {
    if (!selectedModel) {
      toast.error('Please select a model');
      return;
    }
    
    // Store selections in sessionStorage
    sessionStorage.setItem('modelType', modelType);
    sessionStorage.setItem('modelCategory', modelCategory || '');
    sessionStorage.setItem('modelName', selectedModel);
    
    navigate('/upload-dataset');
  };

  return (
    <div className="min-h-screen noise-bg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-12">
          <Button
            data-testid="back-to-home-btn"
            variant="ghost"
            onClick={() => navigate('/')}
            className="mb-6 hover:text-primary transition-colors"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Button>
          <h1 className="text-4xl sm:text-5xl font-heading font-bold mb-4" data-testid="page-title">
            Select Your <span className="text-primary">Model</span>
          </h1>
          <p className="text-lg text-textMuted">Choose the type of machine learning model for your task</p>
        </div>

        {/* Step 1: Model Type Selection */}
        {!modelType && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            <div
              data-testid="model-type-supervised"
              onClick={() => setModelType('supervised')}
              className="glass rounded-lg p-8 cursor-pointer hover:border-primary transition-all duration-300 hover:shadow-glow group"
              style={{
                backgroundImage: `url('https://images.unsplash.com/photo-1546557016-950084dd8399?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NDk1Nzh8MHwxfHNlYXJjaHw0fHxmdXR1cmlzdGljJTIwaHVkJTIwaW50ZXJmYWNlJTIwZWxlbWVudHMlMjBkYXJrJTIwbW9kZXxlbnwwfHx8fDE3NjY2Nzg0OTB8MA&ixlib=rb-4.1.0&q=85')`,
                backgroundSize: 'cover',
                backgroundPosition: 'center',
              }}
            >
              <div className="relative z-10 bg-surface/90 backdrop-blur-sm p-8 rounded-lg">
                <Sparkles className="h-12 w-12 text-primary mb-4 group-hover:scale-110 transition-transform" />
                <h3 className="text-2xl font-heading font-bold mb-2">Supervised Learning</h3>
                <p className="text-textMuted mb-4">
                  Train models with labeled data for classification or regression tasks
                </p>
                <ul className="space-y-2 text-sm text-textMuted">
                  <li>• Classification (Logistic Regression, Random Forest, SVM, XGBoost)</li>
                  <li>• Regression (Linear, Decision Tree, Random Forest, SVR)</li>
                </ul>
              </div>
            </div>

            <div
              data-testid="model-type-unsupervised"
              onClick={() => setModelType('unsupervised')}
              className="glass rounded-lg p-8 cursor-pointer hover:border-secondary transition-all duration-300 hover:shadow-glow-purple group"
              style={{
                backgroundImage: `url('https://images.unsplash.com/photo-1647356191320-d7a1f80ca777?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTY2NjZ8MHwxfHNlYXJjaHwxfHxhYnN0cmFjdCUyMG5ldXJhbCUyMG5ldHdvcmslMjBkYXRhJTIwZmxvdyUyMGRhcmslMjBiYWNrZ3JvdW5kfGVufDB8fHx8MTc2NjY3ODQ4OHww&ixlib=rb-4.1.0&q=85')`,
                backgroundSize: 'cover',
                backgroundPosition: 'center',
              }}
            >
              <div className="relative z-10 bg-surface/90 backdrop-blur-sm p-8 rounded-lg">
                <Layers className="h-12 w-12 text-secondary mb-4 group-hover:scale-110 transition-transform" />
                <h3 className="text-2xl font-heading font-bold mb-2">Unsupervised Learning</h3>
                <p className="text-textMuted mb-4">
                  Discover patterns in unlabeled data through clustering and dimensionality reduction
                </p>
                <ul className="space-y-2 text-sm text-textMuted">
                  <li>• Clustering (K-Means, DBSCAN, Hierarchical)</li>
                  <li>• Dimensionality Reduction (PCA)</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Step 2: Category Selection (Supervised only) */}
        {modelType === 'supervised' && !modelCategory && (
          <div>
            <Button
              data-testid="back-to-type-selection-btn"
              variant="ghost"
              onClick={() => setModelType(null)}
              className="mb-6"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Change Model Type
            </Button>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div
                data-testid="model-category-classification"
                onClick={() => setModelCategory('classification')}
                className="glass rounded-lg p-8 cursor-pointer hover:border-primary transition-all duration-300"
              >
                <h3 className="text-2xl font-heading font-bold mb-2">Classification</h3>
                <p className="text-textMuted mb-4">Predict discrete categories or classes</p>
                <div className="space-y-2">
                  {models?.supervised?.classification?.map((model, idx) => (
                    <div key={idx} className="text-sm text-textMuted">• {model}</div>
                  ))}
                </div>
              </div>

              <div
                data-testid="model-category-regression"
                onClick={() => setModelCategory('regression')}
                className="glass rounded-lg p-8 cursor-pointer hover:border-primary transition-all duration-300"
              >
                <h3 className="text-2xl font-heading font-bold mb-2">Regression</h3>
                <p className="text-textMuted mb-4">Predict continuous numerical values</p>
                <div className="space-y-2">
                  {models?.supervised?.regression?.map((model, idx) => (
                    <div key={idx} className="text-sm text-textMuted">• {model}</div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Model Selection */}
        {((modelType === 'supervised' && modelCategory) || modelType === 'unsupervised') && (
          <div>
            <Button
              data-testid="back-to-category-btn"
              variant="ghost"
              onClick={() => {
                if (modelType === 'supervised') {
                  setModelCategory(null);
                } else {
                  setModelType(null);
                }
                setSelectedModel(null);
              }}
              className="mb-6"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>
            
            <h2 className="text-2xl font-heading font-bold mb-6">
              Choose a {modelType === 'supervised' ? modelCategory : 'clustering'} Algorithm
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              {(modelType === 'supervised' 
                ? models?.supervised?.[modelCategory] 
                : models?.unsupervised
              )?.map((model, idx) => (
                <div
                  key={idx}
                  data-testid={`model-card-${model.toLowerCase().replace(/\s+/g, '-')}`}
                  onClick={() => setSelectedModel(model)}
                  className={`glass rounded-lg p-6 cursor-pointer transition-all duration-300 ${
                    selectedModel === model
                      ? 'border-primary shadow-glow'
                      : 'hover:border-primary/50'
                  }`}
                >
                  <h3 className="text-lg font-heading font-bold mb-2">{model}</h3>
                  <p className="text-sm text-textMuted">Click to select this model</p>
                </div>
              ))}
            </div>

            <div className="flex justify-end">
              <Button
                data-testid="next-to-upload-btn"
                onClick={handleNext}
                disabled={!selectedModel}
                className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-6 shadow-glow disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next: Upload Dataset
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}