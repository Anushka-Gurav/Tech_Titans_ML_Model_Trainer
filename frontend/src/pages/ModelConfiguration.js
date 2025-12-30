import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { ArrowLeft, Loader2, Play } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

export default function ModelConfiguration() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [columns, setColumns] = useState([]);
  const [targetColumn, setTargetColumn] = useState('');
  const [parameters, setParameters] = useState({});
  const [paramConfig, setParamConfig] = useState({});
  const [cleaning, setCleaning] = useState(false);
  const [cleaned, setCleaned] = useState(false);

  const modelType = sessionStorage.getItem('modelType');
  const modelCategory = sessionStorage.getItem('modelCategory');
  const modelName = sessionStorage.getItem('modelName');
  const datasetId = sessionStorage.getItem('datasetId');

  useEffect(() => {
    if (!modelType || !modelName || !datasetId) {
      navigate('/select-model');
      return;
    }
    fetchColumns();
    fetchParameters();
  }, []);

  const fetchColumns = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/dataset/${datasetId}/columns`);
      setColumns(response.data.columns);
    } catch (error) {
      toast.error('Failed to load dataset columns');
    }
  };

  const fetchParameters = async () => {
    try {
      let url = `${BACKEND_URL}/api/models/parameters/${modelType}/${modelName}`;
      if (modelCategory) {
        url += `?model_category=${modelCategory}`;
      }
      const response = await axios.get(url);
      setParamConfig(response.data);
      
      // Initialize parameters with defaults
      const defaultParams = {};
      Object.keys(response.data).forEach(key => {
        defaultParams[key] = response.data[key].default;
      });
      setParameters(defaultParams);
    } catch (error) {
      toast.error('Failed to load model parameters');
    }
  };

  const handleCleanData = async () => {
    setCleaning(true);
    try {
      await axios.post(`${BACKEND_URL}/api/dataset/clean`, {
        dataset_id: datasetId,
        target_column: modelType === 'supervised' ? targetColumn : null,
      });
      setCleaned(true);
      toast.success('Data cleaned successfully!');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Data cleaning failed');
    } finally {
      setCleaning(false);
    }
  };

  const handleTrain = async () => {
    if (modelType === 'supervised' && !targetColumn) {
      toast.error('Please select a target column');
      return;
    }

    if (!cleaned) {
      toast.error('Please clean the data first');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/model/train`, {
        dataset_id: datasetId,
        model_type: modelType,
        model_category: modelCategory,
        model_name: modelName,
        parameters,
        target_column: modelType === 'supervised' ? targetColumn : null,
      });
      
      sessionStorage.setItem('jobId', response.data.job_id);
      navigate('/training');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Training failed to start');
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen noise-bg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <Button
          data-testid="back-to-upload-btn"
          variant="ghost"
          onClick={() => navigate('/upload-dataset')}
          className="mb-6"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>

        <h1 className="text-4xl sm:text-5xl font-heading font-bold mb-4" data-testid="page-title">
          Configure <span className="text-primary">{modelName}</span>
        </h1>
        <p className="text-lg text-textMuted mb-8">
          Model Type: {modelType} {modelCategory && `(${modelCategory})`}
        </p>

        <div className="space-y-8">
          {/* Target Column Selection (Supervised only) */}
          {modelType === 'supervised' && (
            <div className="glass rounded-lg p-8">
              <h2 className="text-2xl font-heading font-bold mb-4">Select Target Column</h2>
              <p className="text-textMuted mb-4">
                Choose the column you want to predict
              </p>
              <Select value={targetColumn} onValueChange={setTargetColumn}>
                <SelectTrigger data-testid="target-column-select" className="bg-surface border-border text-textMain">
                  <SelectValue placeholder="Select target column" />
                </SelectTrigger>
                <SelectContent>
                  {columns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Data Cleaning */}
          <div className="glass rounded-lg p-8">
            <h2 className="text-2xl font-heading font-bold mb-4">Data Cleaning</h2>
            <p className="text-textMuted mb-4">
              Automatically handle missing values, duplicates, encoding, scaling, and outliers
            </p>
            <div className="space-y-2 text-sm text-textMuted mb-4">
              <div>• Remove duplicate rows</div>
              <div>• Fill missing values (median for numerical, mode for categorical)</div>
              <div>• Encode categorical variables</div>
              <div>• Scale numerical features</div>
            </div>
            <Button
              data-testid="clean-data-btn"
              onClick={handleCleanData}
              disabled={cleaning || cleaned}
              className={`${
                cleaned
                  ? 'bg-success hover:bg-success/90'
                  : 'bg-primary hover:bg-primary/90'
              } text-primary-foreground shadow-glow`}
            >
              {cleaning ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Cleaning...
                </>
              ) : cleaned ? (
                'Data Cleaned ✓'
              ) : (
                'Clean Data'
              )}
            </Button>
          </div>

          {/* Model Parameters */}
          {Object.keys(paramConfig).length > 0 && (
            <div className="glass rounded-lg p-8">
              <h2 className="text-2xl font-heading font-bold mb-4">Model Parameters</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {Object.entries(paramConfig).map(([key, config]) => (
                  <div key={key}>
                    <Label htmlFor={key} className="mb-2 block capitalize">
                      {key.replace(/_/g, ' ')}
                    </Label>
                    {config.type === 'select' ? (
                      <Select
                        value={String(parameters[key])}
                        onValueChange={(value) => setParameters({ ...parameters, [key]: value })}
                      >
                        <SelectTrigger data-testid={`param-${key}-select`} className="bg-surface border-border text-textMain">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {config.options.map((opt) => (
                            <SelectItem key={opt} value={opt}>
                              {opt}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    ) : (
                      <Input
                        id={key}
                        data-testid={`param-${key}-input`}
                        type="number"
                        value={parameters[key] || ''}
                        onChange={(e) =>
                          setParameters({
                            ...parameters,
                            [key]: e.target.value ? parseFloat(e.target.value) : null,
                          })
                        }
                        placeholder={`Default: ${config.default || 'None'}`}
                        className="bg-surface border-border"
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Start Training */}
          <div className="flex justify-end">
            <Button
              data-testid="start-training-btn"
              onClick={handleTrain}
              disabled={loading || !cleaned}
              className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-6 shadow-glow text-lg"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Starting...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-5 w-5" />
                  Start Training
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}