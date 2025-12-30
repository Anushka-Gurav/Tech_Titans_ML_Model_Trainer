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

export default function ComparisonConfiguration() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [columns, setColumns] = useState([]);
  const [targetColumn, setTargetColumn] = useState('');
  const [model1Params, setModel1Params] = useState({});
  const [model2Params, setModel2Params] = useState({});
  const [model1Config, setModel1Config] = useState({});
  const [model2Config, setModel2Config] = useState({});
  const [cleaning, setCleaning] = useState(false);
  const [cleaned, setCleaned] = useState(false);

  const modelCategory = sessionStorage.getItem('modelCategory');
  const model1Name = sessionStorage.getItem('model1');
  const model2Name = sessionStorage.getItem('model2');
  const datasetId = sessionStorage.getItem('datasetId');

  useEffect(() => {
    if (!model1Name || !model2Name || !datasetId) {
      navigate('/compare-models');
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
      const url1 = `${BACKEND_URL}/api/models/parameters/supervised/${model1Name}?model_category=${modelCategory}`;
      const url2 = `${BACKEND_URL}/api/models/parameters/supervised/${model2Name}?model_category=${modelCategory}`;
      
      const [res1, res2] = await Promise.all([
        axios.get(url1),
        axios.get(url2)
      ]);
      
      setModel1Config(res1.data);
      setModel2Config(res2.data);
      
      // Initialize with defaults
      const defaults1 = {};
      const defaults2 = {};
      Object.keys(res1.data).forEach(key => {
        defaults1[key] = res1.data[key].default;
      });
      Object.keys(res2.data).forEach(key => {
        defaults2[key] = res2.data[key].default;
      });
      setModel1Params(defaults1);
      setModel2Params(defaults2);
    } catch (error) {
      toast.error('Failed to load model parameters');
    }
  };

  const handleCleanData = async () => {
    setCleaning(true);
    try {
      await axios.post(`${BACKEND_URL}/api/dataset/clean`, {
        dataset_id: datasetId,
        target_column: targetColumn,
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
    if (!targetColumn) {
      toast.error('Please select a target column');
      return;
    }

    if (!cleaned) {
      toast.error('Please clean the data first');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/model/compare`, {
        dataset_id: datasetId,
        model_category: modelCategory,
        model1_name: model1Name,
        model2_name: model2Name,
        model1_parameters: model1Params,
        model2_parameters: model2Params,
        target_column: targetColumn,
      });
      
      sessionStorage.setItem('jobId', response.data.job_id);
      sessionStorage.setItem('isComparison', 'true');
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
          variant="ghost"
          onClick={() => navigate('/upload-dataset')}
          className="mb-6"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>

        <h1 className="text-4xl sm:text-5xl font-heading font-bold mb-4">
          Configure <span className="text-primary">Comparison</span>
        </h1>
        <p className="text-lg text-textMuted mb-8">
          Comparing: {model1Name} vs {model2Name}
        </p>

        <div className="space-y-8">
          {/* Target Column */}
          <div className="glass rounded-lg p-8">
            <h2 className="text-2xl font-heading font-bold mb-4">Select Target Column</h2>
            <p className="text-textMuted mb-4">Choose the column you want to predict</p>
            <Select value={targetColumn} onValueChange={setTargetColumn}>
              <SelectTrigger className="bg-surface border-border text-textMain">
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

          {/* Data Cleaning */}
          <div className="glass rounded-lg p-8">
            <h2 className="text-2xl font-heading font-bold mb-4">Data Cleaning</h2>
            <p className="text-textMuted mb-4">
              Automatically handle missing values, duplicates, encoding, scaling, and outliers
            </p>
            <Button
              onClick={handleCleanData}
              disabled={cleaning || cleaned}
              className={`${
                cleaned ? 'bg-success hover:bg-success/90' : 'bg-primary hover:bg-primary/90'
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

          {/* Model 1 Parameters */}
          {Object.keys(model1Config).length > 0 && (
            <div className="glass rounded-lg p-8 border-l-4 border-primary">
              <h2 className="text-2xl font-heading font-bold mb-4">
                {model1Name} Parameters
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {Object.entries(model1Config).map(([key, config]) => (
                  <div key={key}>
                    <Label htmlFor={`m1-${key}`} className="mb-2 block capitalize">
                      {key.replace(/_/g, ' ')}
                    </Label>
                    {config.type === 'select' ? (
                      <Select
                        value={String(model1Params[key])}
                        onValueChange={(value) => setModel1Params({ ...model1Params, [key]: value })}
                      >
                        <SelectTrigger className="bg-surface border-border text-textMain">
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
                        id={`m1-${key}`}
                        type="number"
                        value={model1Params[key] || ''}
                        onChange={(e) =>
                          setModel1Params({
                            ...model1Params,
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

          {/* Model 2 Parameters */}
          {Object.keys(model2Config).length > 0 && (
            <div className="glass rounded-lg p-8 border-l-4 border-secondary">
              <h2 className="text-2xl font-heading font-bold mb-4">
                {model2Name} Parameters
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {Object.entries(model2Config).map(([key, config]) => (
                  <div key={key}>
                    <Label htmlFor={`m2-${key}`} className="mb-2 block capitalize">
                      {key.replace(/_/g, ' ')}
                    </Label>
                    {config.type === 'select' ? (
                      <Select
                        value={String(model2Params[key])}
                        onValueChange={(value) => setModel2Params({ ...model2Params, [key]: value })}
                      >
                        <SelectTrigger className="bg-surface border-border text-textMain">
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
                        id={`m2-${key}`}
                        type="number"
                        value={model2Params[key] || ''}
                        onChange={(e) =>
                          setModel2Params({
                            ...model2Params,
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
                  Start Comparison
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
