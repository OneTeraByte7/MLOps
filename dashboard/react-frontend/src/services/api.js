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
export default apiService
