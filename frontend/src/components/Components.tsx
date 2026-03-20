import { Activity, Brain, Sparkles, Stethoscope } from 'lucide-react';

export interface PredictionResult {
  condition: string;
  predicted_procedure: string;
  confidence: number;
}

export function Header() {
  return (
    <header className="text-center mb-8">
      <div className="flex items-center justify-center gap-3 mb-2">
        <div className="relative">
          <div className="absolute inset-0 bg-blue-400 rounded-full animate-pulse-ring"></div>
          <Stethoscope className="w-12 h-12 text-white relative z-10" />
        </div>
      </div>
      <h1 className="text-4xl font-bold text-white mb-2">
        MedPredict
      </h1>
      <p className="text-blue-100 text-lg flex items-center justify-center gap-2">
        <Brain className="w-5 h-5" />
        AI-Powered Medical Procedure Prediction
      </p>
    </header>
  );
}

export function PredictionCard({ result }: { result: PredictionResult }) {
  const confidenceColor = result.confidence >= 80 
    ? 'text-emerald-300' 
    : result.confidence >= 60 
    ? 'text-yellow-300' 
    : 'text-orange-300';

  return (
    <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 animate-fade-in border border-white/20">
      <div className="flex items-start gap-4">
        <div className="bg-gradient-to-br from-emerald-400 to-teal-500 rounded-xl p-3">
          <Activity className="w-6 h-6 text-white" />
        </div>
        <div className="flex-1">
          <p className="text-blue-200 text-sm mb-1">Predicted Procedure</p>
          <h3 className="text-white text-xl font-semibold mb-3">
            {result.predicted_procedure}
          </h3>
          <div className="flex items-center gap-2">
            <Sparkles className={`w-4 h-4 ${confidenceColor}`} />
            <span className={`font-medium ${confidenceColor}`}>
              {result.confidence}% Confidence
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function ConditionBadge({ condition }: { condition: string }) {
  return (
    <div className="inline-flex items-center gap-2 bg-white/20 rounded-full px-4 py-2 text-white">
      <span className="text-sm">Input:</span>
      <span className="font-medium">{condition}</span>
    </div>
  );
}

export function FeatureCard({ 
  icon: Icon, 
  title, 
  description 
}: { 
  icon: React.ElementType; 
  title: string; 
  description: string; 
}) {
  return (
    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/10 hover:bg-white/20 transition-colors">
      <Icon className="w-6 h-6 text-blue-300 mb-2" />
      <h3 className="text-white font-medium mb-1">{title}</h3>
      <p className="text-blue-200 text-sm">{description}</p>
    </div>
  );
}
