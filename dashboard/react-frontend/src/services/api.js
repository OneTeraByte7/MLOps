import axios from 'axios'

const API_BASE_URL = '/api'

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// API Service
export const apiService = {
  // Health check
  healthCheck: async () => {
    try {
      const response = await api.get('/health')
      return response.data
    } catch (error) {
      console.error('Health check failed:', error)
      throw error
    }
  },

  // Get model info
  getModelInfo: async () => {
    try {
      const response = await api.get('/model_info')
      return response.data
    } catch (error) {
      console.error('Failed to get model info:', error)
      throw error
    }
  },

  // Make prediction
  predict: async (customerData) => {
    try {
      const response = await api.post('/predict', customerData)
      return response.data
    } catch (error) {
      console.error('Prediction failed:', error)
      throw error
    }
  },

  // Batch prediction
  batchPredict: async (customers) => {
    try {
      const response = await api.post('/batch_predict', { customers })
      return response.data
    } catch (error) {
      console.error('Batch prediction failed:', error)
      throw error
    }
  },

  // Get explanation
  getExplanation: async (customerData) => {
    try {
      const response = await api.post('/explain', customerData)
      return response.data
    } catch (error) {
      console.error('Explanation failed:', error)
      throw error
    }
  },

  // Get metrics (Prometheus)
  getMetrics: async () => {
    try {
      const response = await api.get('/metrics')
      return response.data
    } catch (error) {
      console.error('Failed to get metrics:', error)
      throw error
    }
  },
}

// Mock data generators for development
export const mockData = {
  generatePredictions: (count = 1000) => {
    const predictions = []
    for (let i = 0; i < count; i++) {
      predictions.push({
        timestamp: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000),
        customer_id: `CUST_${String(i).padStart(6, '0')}`,
        churn_probability: Math.random() * 0.3 + (Math.random() > 0.75 ? 0.5 : 0.1),
        prediction: Math.random() > 0.75 ? 'Yes' : 'No',
        risk_level: Math.random() > 0.85 ? 'High' : Math.random() > 0.5 ? 'Medium' : 'Low',
        model_version: Math.random() > 0.6 ? 'v1.0.0' : 'v1.1.0',
      })
    }
    return predictions
  },

  generateMLFlowRuns: (count = 20) => {
    const runs = []
    for (let i = 0; i < count; i++) {
      runs.push({
        run_id: `run_${i}`,
        start_time: new Date(Date.now() - i * 24 * 60 * 60 * 1000),
        'metrics.test_auc': 0.85 + Math.random() * 0.1,
        'metrics.test_f1': 0.80 + Math.random() * 0.1,
        'metrics.test_precision': 0.82 + Math.random() * 0.1,
        'metrics.test_recall': 0.78 + Math.random() * 0.1,
        'params.max_depth': Math.floor(Math.random() * 5) + 3,
        'params.learning_rate': 0.01 + Math.random() * 0.09,
      })
    }
    return runs
  },

  generateDriftReport: () => {
    const features = [
      'monthly_revenue',
      'logins_per_month',
      'feature_usage_depth',
      'support_tickets',
      'account_age_days',
      'nps_score',
      'payment_delays',
      'days_since_last_login',
    ]

    const featureDrift = {}
    features.forEach((feature) => {
      const psi = Math.random() * 0.3
      featureDrift[feature] = {
        psi,
        drift_detected: psi > 0.1,
        mean_shift: (Math.random() - 0.5) * 0.3,
        current_mean: Math.random() * 100,
        reference_mean: Math.random() * 100,
      }
    })

    return {
      drift_status: {
        overall_status: 'warning',
        alerts: [
          {
            severity: 'high',
            message: 'Significant drift detected in monthly_revenue',
          },
          {
            severity: 'medium',
            message: 'Moderate drift in logins_per_month',
          },
        ],
        recommendations: [
          'Consider retraining the model with recent data',
          'Investigate changes in customer behavior patterns',
          'Review feature engineering pipeline',
        ],
      },
      drifted_features_count: Object.values(featureDrift).filter((f) => f.drift_detected).length,
      feature_drift: featureDrift,
      label_drift: {
        churn_rate_change: 0.023,
      },
      model_performance: {
        auc: 0.9234,
      },
    }
  },
}

export { api }
export default apiService
