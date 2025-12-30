import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Button } from '../components/ui/button';
import { ArrowLeft, ArrowRight, Sparkles } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

export default function ModelComparison() {
  const navigate = useNavigate();
  const [modelCategory, setModelCategory] = useState(null);
  const [selectedModels, setSelectedModels] = useState([]);
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

  const handleModelToggle = (modelName) => {
    if (selectedModels.includes(modelName)) {
      setSelectedModels(selectedModels.filter(m => m !== modelName));
    } else {
      if (selectedModels.length < 2) {
        setSelectedModels([...selectedModels, modelName]);
      } else {
        toast.error('You can only select 2 models for comparison');
      }
    }
  };

  const handleNext = () => {
    if (selectedModels.length !== 2) {
      toast.error('Please select exactly 2 models to compare');
      return;
    }
    
    sessionStorage.setItem('comparisonMode', 'true');
    sessionStorage.setItem('modelCategory', modelCategory);
    sessionStorage.setItem('model1', selectedModels[0]);
    sessionStorage.setItem('model2', selectedModels[1]);
    
    navigate('/upload-dataset');
  };

  return (
    <div className="min-h-screen noise-bg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
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
          Compare <span className="text-primary">Models</span>
        </h1>
        <p className="text-lg text-textMuted mb-8">Select 2 models to compare their performance</p>

        {!modelCategory && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div
              data-testid="category-classification"
              onClick={() => setModelCategory('classification')}
              className="glass rounded-lg p-8 cursor-pointer hover:border-primary transition-all duration-300 hover:shadow-glow"
            >
              <Sparkles className="h-12 w-12 text-primary mb-4" />
              <h3 className="text-2xl font-heading font-bold mb-2">Classification Models</h3>
              <p className="text-textMuted mb-4">Compare classification algorithms</p>
              <div className="space-y-2">
                {models?.supervised?.classification?.map((model, idx) => (
                  <div key={idx} className="text-sm text-textMuted">• {model}</div>
                ))}
              </div>
            </div>

            <div
              data-testid="category-regression"
              onClick={() => setModelCategory('regression')}
              className="glass rounded-lg p-8 cursor-pointer hover:border-primary transition-all duration-300 hover:shadow-glow"
            >
              <Sparkles className="h-12 w-12 text-secondary mb-4" />
              <h3 className="text-2xl font-heading font-bold mb-2">Regression Models</h3>
              <p className="text-textMuted mb-4">Compare regression algorithms</p>
              <div className="space-y-2">
                {models?.supervised?.regression?.map((model, idx) => (
                  <div key={idx} className="text-sm text-textMuted">• {model}</div>
                ))}
              </div>
            </div>
          </div>
        )}

        {modelCategory && (
          <div>
            <Button
              variant="ghost"
              onClick={() => {
                setModelCategory(null);
                setSelectedModels([]);
              }}
              className="mb-6"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Change Category
            </Button>
            
            <div className="glass rounded-lg p-6 mb-6">
              <div className="flex items-center justify-between">
                <p className="text-textMuted">Selected Models: {selectedModels.length} / 2</p>
                {selectedModels.length === 2 && (
                  <div className="flex gap-2">
                    <span className="px-3 py-1 bg-primary/20 text-primary rounded-md text-sm">
                      {selectedModels[0]}
                    </span>
                    <span className="text-textMuted">vs</span>
                    <span className="px-3 py-1 bg-secondary/20 text-secondary rounded-md text-sm">
                      {selectedModels[1]}
                    </span>
                  </div>
                )}
              </div>
            </div>
            
            <h2 className="text-2xl font-heading font-bold mb-6">
              Select 2 {modelCategory} Models
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              {models?.supervised?.[modelCategory]?.map((model, idx) => {
                const isSelected = selectedModels.includes(model);
                const selectionOrder = selectedModels.indexOf(model) + 1;
                
                return (
                  <div
                    key={idx}
                    data-testid={`model-card-${model.toLowerCase().replace(/\s+/g, '-')}`}
                    onClick={() => handleModelToggle(model)}
                    className={`glass rounded-lg p-6 cursor-pointer transition-all duration-300 relative ${
                      isSelected
                        ? selectionOrder === 1
                          ? 'border-primary shadow-glow'
                          : 'border-secondary shadow-glow-purple'
                        : 'hover:border-primary/50'
                    }`}
                  >
                    {isSelected && (
                      <div className={`absolute top-3 right-3 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                        selectionOrder === 1 ? 'bg-primary text-primary-foreground' : 'bg-secondary text-secondary-foreground'
                      }`}>
                        {selectionOrder}
                      </div>
                    )}
                    <h3 className="text-lg font-heading font-bold mb-2">{model}</h3>
                    <p className="text-sm text-textMuted">
                      {isSelected ? 'Selected' : 'Click to select'}
                    </p>
                  </div>
                );
              })}
            </div>

            <div className="flex justify-end">
              <Button
                data-testid="next-to-upload-btn"
                onClick={handleNext}
                disabled={selectedModels.length !== 2}
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