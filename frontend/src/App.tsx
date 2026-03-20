import { useState, useCallback } from 'react';
import { Send, Loader2, Brain, Zap, Clock } from 'lucide-react';
import { Header, PredictionCard, ConditionBadge, FeatureCard } from './components/Components';
import { predictProcedure as apiPredict } from './api/prediction';

function App() {
  const [input, setInput] = useState('');
  const [result, setResult] = useState<{ condition: string; predicted_procedure: string; confidence: number } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await apiPredict({ condition: input.trim() });
      setResult(response);
    } catch {
      setError('Failed to get prediction. Please ensure the backend is running.');
    } finally {
      setIsLoading(false);
    }
  }, [input]);

  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-2xl mx-auto">
        <Header />

        <div className="bg-white/20 backdrop-blur-xl rounded-3xl p-6 shadow-2xl border border-white/30 mb-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="condition" className="block text-white text-sm font-medium mb-2">
                Enter Medical Condition
              </label>
              <input
                id="condition"
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="e.g., Heart Disease, Diabetes, Stroke..."
                className="w-full px-4 py-3 rounded-xl bg-white/90 text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-400 transition-all"
                disabled={isLoading}
              />
            </div>
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="w-full py-3 px-6 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 shadow-lg"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Predicting...
                </>
              ) : (
                <>
                  <Send className="w-5 h-5" />
                  Get Prediction
                </>
              )}
            </button>
          </form>

          {error && (
            <div className="mt-4 p-4 bg-red-500/20 border border-red-400/30 rounded-xl text-red-200 text-sm">
              {error}
            </div>
          )}
        </div>

        {result && (
          <div className="space-y-4 animate-fade-in">
            <ConditionBadge condition={result.condition} />
            <PredictionCard result={result} />
          </div>
        )}

        {!result && !isLoading && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <FeatureCard
              icon={Brain}
              title="Semantic Analysis"
              description="Understands medical terminology"
            />
            <FeatureCard
              icon={Zap}
              title="High Accuracy"
              description="Powered by neural networks"
            />
            <FeatureCard
              icon={Clock}
              title="Fast Results"
              description="Instant predictions"
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
