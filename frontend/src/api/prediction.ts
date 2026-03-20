const API_URL = (import.meta as any)?.env?.VITE_API_URL || '/api';

export interface PredictionRequest {
  condition: string;
}

export interface PredictionResponse {
  condition: string;
  predicted_procedure: string;
  confidence: number;
}

export async function predictProcedure(request: PredictionRequest): Promise<PredictionResponse> {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error('Prediction failed');
  }

  return response.json();
}

export async function checkHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_URL}/health`);
  return response.json();
}
