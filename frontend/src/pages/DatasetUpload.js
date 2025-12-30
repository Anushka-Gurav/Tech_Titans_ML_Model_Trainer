import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { ArrowLeft, ArrowRight, Upload, Database, Loader2 } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

export default function DatasetUpload() {
  const navigate = useNavigate();
  const [uploading, setUploading] = useState(false);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [file, setFile] = useState(null);
  const [kaggleDataset, setKaggleDataset] = useState('');
  const [kaggleUsername, setKaggleUsername] = useState('');
  const [kaggleKey, setKaggleKey] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const ext = selectedFile.name.split('.').pop().toLowerCase();
      if (!['csv', 'xlsx', 'xls'].includes(ext)) {
        toast.error('Please upload a CSV or Excel file');
        return;
      }
      setFile(selectedFile);
    }
  };

  const handleFileUpload = async () => {
    if (!file) {
      toast.error('Please select a file');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${BACKEND_URL}/api/dataset/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setDatasetInfo(response.data);
      sessionStorage.setItem('datasetId', response.data.dataset_id);
      toast.success('Dataset uploaded successfully!');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleKaggleUpload = async () => {
    if (!kaggleDataset || !kaggleUsername || !kaggleKey) {
      toast.error('Please fill all Kaggle credentials');
      return;
    }

    setUploading(true);
    try {
      const response = await axios.post(
        `${BACKEND_URL}/api/dataset/kaggle?dataset_name=${encodeURIComponent(kaggleDataset)}&kaggle_username=${kaggleUsername}&kaggle_key=${kaggleKey}`
      );
      setDatasetInfo(response.data);
      sessionStorage.setItem('datasetId', response.data.dataset_id);
      toast.success('Kaggle dataset fetched successfully!');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Kaggle fetch failed');
    } finally {
      setUploading(false);
    }
  };

  const handleNext = () => {
    if (!datasetInfo) {
      toast.error('Please upload a dataset first');
      return;
    }
    
    // Check if in comparison mode
    const isComparison = sessionStorage.getItem('comparisonMode') === 'true';
    if (isComparison) {
      navigate('/configure-comparison');
    } else {
      navigate('/configure-model');
    }
  };

  return (
    <div className="min-h-screen noise-bg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <Button
          data-testid="back-to-model-selection-btn"
          variant="ghost"
          onClick={() => navigate('/select-model')}
          className="mb-6"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>

        <h1 className="text-4xl sm:text-5xl font-heading font-bold mb-4" data-testid="page-title">
          Upload Your <span className="text-primary">Dataset</span>
        </h1>
        <p className="text-lg text-textMuted mb-8">Upload a local file or fetch from Kaggle</p>

        <Tabs defaultValue="local" className="w-full">
          <TabsList className="glass border border-border" data-testid="upload-tabs">
            <TabsTrigger value="local" className="data-[state=active]:bg-primary/20" data-testid="tab-local-upload">
              <Upload className="mr-2 h-4 w-4" />
              Local Upload
            </TabsTrigger>
            <TabsTrigger value="kaggle" className="data-[state=active]:bg-primary/20" data-testid="tab-kaggle-upload">
              <Database className="mr-2 h-4 w-4" />
              Kaggle Dataset
            </TabsTrigger>
          </TabsList>

          <TabsContent value="local" className="mt-8">
            <div className="glass rounded-lg p-8">
              <Label htmlFor="file-upload" className="text-lg font-heading mb-4 block">
                Select File (CSV or Excel)
              </Label>
              <div className="flex gap-4">
                <Input
                  id="file-upload"
                  data-testid="file-upload-input"
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  onChange={handleFileChange}
                  className="flex-1 bg-surface border-border"
                />
                <Button
                  data-testid="upload-file-btn"
                  onClick={handleFileUpload}
                  disabled={!file || uploading}
                  className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-glow"
                >
                  {uploading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>Upload</>                  )}
                </Button>
              </div>
              {file && <p className="text-sm text-textMuted mt-2">Selected: {file.name}</p>}
            </div>
          </TabsContent>

          <TabsContent value="kaggle" className="mt-8">
            <div className="glass rounded-lg p-8 space-y-6">
              <div>
                <Label htmlFor="kaggle-dataset" className="text-lg font-heading mb-2 block">
                  Kaggle Dataset Name
                </Label>
                <Input
                  id="kaggle-dataset"
                  data-testid="kaggle-dataset-input"
                  placeholder="e.g., username/dataset-name"
                  value={kaggleDataset}
                  onChange={(e) => setKaggleDataset(e.target.value)}
                  className="bg-surface border-border"
                />
                <p className="text-sm text-textMuted mt-1">
                  Format: owner/dataset-name (e.g., uciml/iris)
                </p>
              </div>

              <div>
                <Label htmlFor="kaggle-username" className="text-lg font-heading mb-2 block">
                  Kaggle Username
                </Label>
                <Input
                  id="kaggle-username"
                  data-testid="kaggle-username-input"
                  placeholder="Your Kaggle username"
                  value={kaggleUsername}
                  onChange={(e) => setKaggleUsername(e.target.value)}
                  className="bg-surface border-border"
                />
              </div>

              <div>
                <Label htmlFor="kaggle-key" className="text-lg font-heading mb-2 block">
                  Kaggle API Key
                </Label>
                <Input
                  id="kaggle-key"
                  data-testid="kaggle-key-input"
                  type="password"
                  placeholder="Your Kaggle API key"
                  value={kaggleKey}
                  onChange={(e) => setKaggleKey(e.target.value)}
                  className="bg-surface border-border"
                />
                <p className="text-sm text-textMuted mt-1">
                  Get your API key from Kaggle Account Settings
                </p>
              </div>

              <Button
                data-testid="fetch-kaggle-btn"
                onClick={handleKaggleUpload}
                disabled={uploading}
                className="w-full bg-primary hover:bg-primary/90 text-primary-foreground shadow-glow"
              >
                {uploading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Fetching...
                  </>
                ) : (
                  <>Fetch Dataset</>
                )}
              </Button>
            </div>
          </TabsContent>
        </Tabs>

        {/* Dataset Preview */}
        {datasetInfo && (
          <div className="mt-8 glass rounded-lg p-8" data-testid="dataset-preview">
            <h2 className="text-2xl font-heading font-bold mb-4">Dataset Preview</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-surface/50 rounded p-4">
                <div className="text-sm text-textMuted">Filename</div>
                <div className="text-lg font-heading font-bold truncate">{datasetInfo.filename}</div>
              </div>
              <div className="bg-surface/50 rounded p-4">
                <div className="text-sm text-textMuted">Rows</div>
                <div className="text-lg font-heading font-bold">{datasetInfo.rows}</div>
              </div>
              <div className="bg-surface/50 rounded p-4">
                <div className="text-sm text-textMuted">Columns</div>
                <div className="text-lg font-heading font-bold">{datasetInfo.columns}</div>
              </div>
              <div className="bg-surface/50 rounded p-4">
                <div className="text-sm text-textMuted">Status</div>
                <div className="text-lg font-heading font-bold text-success">Ready</div>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    {datasetInfo.preview[0] &&
                      Object.keys(datasetInfo.preview[0]).map((key) => (
                        <th key={key} className="text-left p-2 font-heading">
                          {key}
                        </th>
                      ))}
                  </tr>
                </thead>
                <tbody>
                  {datasetInfo.preview.map((row, idx) => (
                    <tr key={idx} className="border-b border-border/50">
                      {Object.values(row).map((val, i) => (
                        <td key={i} className="p-2 text-textMuted">
                          {String(val)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-6 flex justify-end">
              <Button
                data-testid="next-to-configure-btn"
                onClick={handleNext}
                className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-6 shadow-glow"
              >
                Next: Configure Model
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}