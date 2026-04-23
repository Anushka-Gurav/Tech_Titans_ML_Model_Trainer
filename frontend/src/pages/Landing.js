import { useNavigate } from 'react-router-dom';
import { Button } from '../components/ui/button';
import { ArrowRight, Cpu, Database, LineChart } from 'lucide-react';

export default function Landing() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen noise-bg">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div 
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage: `url('https://images.unsplash.com/photo-1746421094550-8ca6940c2f6e?...')`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-background via-background/50 to-background" />
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 sm:py-32">
          <div className="text-center">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-heading font-bold mb-6">
              <span className="text-gradient">ML Training Platform</span>
            </h1>

            <p className="text-lg sm:text-xl text-textMuted max-w-3xl mx-auto mb-8">
              Train machine learning models effortlessly. Upload your data, select an algorithm,
              and get instant results with comprehensive visualizations.
            </p>

            {/* 🔥 BUTTON SECTION */}
            <div className="flex gap-4 justify-center flex-wrap">

              {/* Train Model */}
              <Button
                onClick={() => navigate('/select-model')}
                className="bg-gradient-to-r from-indigo-500 to-purple-600 
                           hover:from-indigo-600 hover:to-purple-700 
                           text-white px-8 py-6 text-lg font-heading 
                           shadow-lg hover:shadow-xl 
                           transition-all duration-300"
              >
                Train Model
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>

              {/* Compare Models */}
              <Button
                onClick={() => navigate('/compare-models')}
                variant="outline"
                className="border-indigo-500 text-indigo-500 
                           hover:bg-indigo-500/10 px-8 py-6 text-lg 
                           font-heading transition-all duration-300"
              >
                Compare Models
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>

              {/* ✅ NEW RIGHT-SIDE BUTTON */}
              <Button
                onClick={() => navigate('/dashboard')}
                className="bg-green-500 hover:bg-green-600 
                           text-white px-8 py-6 text-lg font-heading 
                           shadow-lg hover:shadow-xl 
                           transition-all duration-300"
              >
                View Dashboard
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>

            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">

          <div className="glass rounded-lg p-8 hover:border-indigo-500/50 transition-all duration-300 group">
            <div className="h-12 w-12 rounded-lg bg-indigo-500/10 flex items-center justify-center mb-4">
              <Cpu className="h-6 w-6 text-indigo-500" />
            </div>
            <h3 className="text-xl font-heading font-bold mb-2">Multiple Models</h3>
            <p className="text-textMuted">
              Choose from supervised and unsupervised algorithms.
            </p>
          </div>

          <div className="glass rounded-lg p-8 hover:border-purple-500/50 transition-all duration-300 group">
            <div className="h-12 w-12 rounded-lg bg-purple-500/10 flex items-center justify-center mb-4">
              <Database className="h-6 w-6 text-purple-500" />
            </div>
            <h3 className="text-xl font-heading font-bold mb-2">Auto Data Cleaning</h3>
            <p className="text-textMuted">
              Automatically handle missing values, duplicates, and encoding.
            </p>
          </div>

          <div className="glass rounded-lg p-8 hover:border-yellow-400/50 transition-all duration-300 group">
            <div className="h-12 w-12 rounded-lg bg-yellow-400/10 flex items-center justify-center mb-4">
              <LineChart className="h-6 w-6 text-yellow-400" />
            </div>
            <h3 className="text-xl font-heading font-bold mb-2">Rich Visualizations</h3>
            <p className="text-textMuted">
              Get detailed metrics and model evaluation charts.
            </p>
          </div>

        </div>
      </div>

      {/* How It Works */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <h2 className="text-3xl sm:text-4xl font-heading font-bold text-center mb-12">
          How It <span className="text-indigo-500">Works</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {[
            { step: '01', title: 'Select Model', desc: 'Choose supervised or unsupervised' },
            { step: '02', title: 'Upload Data', desc: 'Upload your dataset' },
            { step: '03', title: 'Configure', desc: 'Set parameters' },
            { step: '04', title: 'Get Results', desc: 'View outputs' },
          ].map((item, idx) => (
            <div key={idx} className="relative">
              <div className="glass rounded-lg p-6">
                <div className="text-5xl font-heading font-bold text-indigo-500/20 mb-2">
                  {item.step}
                </div>
                <h4 className="text-lg font-heading font-bold mb-1">{item.title}</h4>
                <p className="text-sm text-textMuted">{item.desc}</p>
              </div>

              {idx < 3 && (
                <div className="hidden md:block absolute top-1/2 -right-3 transform -translate-y-1/2">
                  <ArrowRight className="h-6 w-6 text-indigo-500/40" />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
