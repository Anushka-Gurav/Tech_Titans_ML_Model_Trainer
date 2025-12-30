import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing';
import ModelSelection from './pages/ModelSelection';
import DatasetUpload from './pages/DatasetUpload';
import ModelConfiguration from './pages/ModelConfiguration';
import TrainingProgress from './pages/TrainingProgress';
import Results from './pages/Results';
import ModelComparison from './pages/ModelComparison';
import ComparisonConfiguration from './pages/ComparisonConfiguration';
import ComparisonResults from './pages/ComparisonResults';
import { Toaster } from 'sonner';

function App() {
  return (
    <div className="App min-h-screen bg-background">
      <Toaster position="top-right" theme="dark" />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/select-model" element={<ModelSelection />} />
          <Route path="/compare-models" element={<ModelComparison />} />
          <Route path="/upload-dataset" element={<DatasetUpload />} />
          <Route path="/configure-model" element={<ModelConfiguration />} />
          <Route path="/configure-comparison" element={<ComparisonConfiguration />} />
          <Route path="/training" element={<TrainingProgress />} />
          <Route path="/results/:jobId" element={<Results />} />
          <Route path="/comparison-results/:jobId" element={<ComparisonResults />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;