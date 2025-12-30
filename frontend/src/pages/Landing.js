import { useState } from 'react';
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
            backgroundImage: `url('https://images.unsplash.com/photo-1746421094550-8ca6940c2f6e?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTY2NjZ8MHwxfHNlYXJjaHw0fHxhYnN0cmFjdCUyMG5ldXJhbCUyMG5ldHdvcmslMjBkYXRhJTIwZmxvdyUyMGRhcmslMjBiYWNrZ3JvdW5kfGVufDB8fHx8MTc2NjY3ODQ4OHww&ixlib=rb-4.1.0&q=85')`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-background via-background/50 to-background" />
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 sm:py-32">
          <div className="text-center">
            <h1 
              className="text-4xl sm:text-5xl lg:text-6xl font-heading font-bold mb-6"
              data-testid="hero-title"
            >
              <span className="text-gradient">ML Training Platform</span>
            </h1>
            <p className="text-lg sm:text-xl text-textMuted max-w-3xl mx-auto mb-8" data-testid="hero-subtitle">
              Train machine learning models effortlessly. Upload your data, select an algorithm,
              and get instant results with comprehensive visualizations.
            </p>
            <div className="flex gap-4 justify-center">
              <Button
                data-testid="get-started-btn"
                onClick={() => navigate('/select-model')}
                className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-6 text-lg font-heading shadow-glow hover:shadow-glow transition-all duration-300"
              >
                Train Model
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button
                data-testid="compare-models-btn"
                onClick={() => navigate('/compare-models')}
                variant="outline"
                className="border-primary text-primary hover:bg-primary/10 px-8 py-6 text-lg font-heading transition-all duration-300"
              >
                Compare Models
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div 
            className="glass rounded-lg p-8 hover:border-primary/50 transition-all duration-300 group"
            data-testid="feature-card-models"
          >
            <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:shadow-glow transition-all duration-300">
              <Cpu className="h-6 w-6 text-primary" />
            </div>
            <h3 className="text-xl font-heading font-bold mb-2">Multiple Models</h3>
            <p className="text-textMuted">
              Choose from supervised (classification, regression) and unsupervised (clustering, PCA) algorithms.
            </p>
          </div>

          <div 
            className="glass rounded-lg p-8 hover:border-secondary/50 transition-all duration-300 group"
            data-testid="feature-card-data"
          >
            <div className="h-12 w-12 rounded-lg bg-secondary/10 flex items-center justify-center mb-4 group-hover:shadow-glow-purple transition-all duration-300">
              <Database className="h-6 w-6 text-secondary" />
            </div>
            <h3 className="text-xl font-heading font-bold mb-2">Auto Data Cleaning</h3>
            <p className="text-textMuted">
              Automatically handle missing values, duplicates, encoding, scaling, and outliers.
            </p>
          </div>

          <div 
            className="glass rounded-lg p-8 hover:border-accent/50 transition-all duration-300 group"
            data-testid="feature-card-viz"
          >
            <div className="h-12 w-12 rounded-lg bg-accent/10 flex items-center justify-center mb-4 group-hover:shadow-[0_0_15px_rgba(250,204,21,0.3)] transition-all duration-300">
              <LineChart className="h-6 w-6 text-accent" />
            </div>
            <h3 className="text-xl font-heading font-bold mb-2">Rich Visualizations</h3>
            <p className="text-textMuted">
              Get detailed metrics, confusion matrices, ROC curves, and more for comprehensive model evaluation.
            </p>
          </div>
        </div>
      </div>

      {/* How It Works */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <h2 className="text-3xl sm:text-4xl font-heading font-bold text-center mb-12">
          How It <span className="text-primary">Works</span>
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {[
            { step: '01', title: 'Select Model', desc: 'Choose supervised or unsupervised' },
            { step: '02', title: 'Upload Data', desc: 'Local files or Kaggle datasets' },
            { step: '03', title: 'Configure', desc: 'Set model parameters' },
            { step: '04', title: 'Get Results', desc: 'View metrics & download model' },
          ].map((item, idx) => (
            <div key={idx} className="relative" data-testid={`how-it-works-step-${idx + 1}`}>
              <div className="glass rounded-lg p-6">
                <div className="text-5xl font-heading font-bold text-primary/20 mb-2">{item.step}</div>
                <h4 className="text-lg font-heading font-bold mb-1">{item.title}</h4>
                <p className="text-sm text-textMuted">{item.desc}</p>
              </div>
              {idx < 3 && (
                <div className="hidden md:block absolute top-1/2 -right-3 transform -translate-y-1/2">
                  <ArrowRight className="h-6 w-6 text-primary/40" />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}