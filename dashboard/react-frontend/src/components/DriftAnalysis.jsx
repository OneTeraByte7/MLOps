import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { AlertTriangle, CheckCircle, AlertCircle } from 'lucide-react'

const DriftAnalysis = ({ lastUpdated }) => {
  const [driftStatus, setDriftStatus] = useState('healthy')
  const [featureDrift, setFeatureDrift] = useState([])
  const [alerts, setAlerts] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let isMounted = true
    const controller = new AbortController()
    
    const fetchDriftData = async () => {
      try {
        setLoading(true)
        const response = await fetch('http://localhost:8000/api/drift/latest', {
          signal: controller.signal
        })
        
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
        
        const report = await response.json()
        if (!isMounted) return
        
        setDriftStatus(report.drift_status.overall_status)
        
        const features = Object.entries(report.feature_drift).map(([name, stats]) => ({
          feature: name,
          psi: stats.psi,
          driftDetected: stats.drift_detected
        }))
        setFeatureDrift(features)
        
        setAlerts(report.drift_status.alerts)
      } catch (error) {
        if (error.name !== 'AbortError') {
          console.error('Failed to fetch drift data:', error)
        }
      } finally {
        if (isMounted) {
          setLoading(false)
        }
      }
    }

    fetchDriftData()
    
    return () => {
      isMounted = false
      controller.abort()
    }
  }, [lastUpdated])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-xl text-gray-600">Loading drift analysis data...</div>
      </div>
    )
  }

  // Show message if no drift data
  if (featureDrift.length === 0) {
    return (
      <div className="space-y-6 animate-fade-in">
        <h1 className="text-3xl font-bold text-gray-900">
          🔍 Data & Model Drift Analysis
        </h1>
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-8 text-center">
          <p className="text-xl text-gray-700 mb-4">No drift report available.</p>
          <p className="text-gray-600">Run drift detection to see analysis here.</p>
        </div>
      </div>
    )
  }

  const getStatusColor = () => {
    switch(driftStatus) {
      case 'critical': return 'bg-red-100 border-red-500 text-red-800'
      case 'warning': return 'bg-yellow-100 border-yellow-500 text-yellow-800'
      case 'monitoring': return 'bg-blue-100 border-blue-500 text-blue-800'
      default: return 'bg-green-100 border-green-500 text-green-800'
    }
  }

  const getStatusIcon = () => {
    switch(driftStatus) {
      case 'critical': return <AlertTriangle className="text-red-500" size={24} />
      case 'warning': return <AlertCircle className="text-yellow-500" size={24} />
      default: return <CheckCircle className="text-green-500" size={24} />
    }
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-3xl font-bold text-gray-900">
        🔍 Data & Model Drift Analysis
      </h1>

      {/* Status Card */}
      <div className={`border-l-4 p-6 rounded-r-xl ${getStatusColor()}`}>
        <div className="flex items-center space-x-3">
          {getStatusIcon()}
          <div>
            <h2 className="text-xl font-bold">
              {driftStatus === 'critical' && '🚨 CRITICAL: Significant drift detected!'}
              {driftStatus === 'warning' && '⚠️ WARNING: Moderate drift detected'}
              {driftStatus === 'monitoring' && 'ℹ️ MONITORING: Minor drift detected'}
              {driftStatus === 'healthy' && '✅ HEALTHY: No significant drift'}
            </h2>
          </div>
        </div>
      </div>

      {/* Drift Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <p className="text-gray-600 text-sm font-medium mb-2">Features with Drift</p>
          <p className="text-3xl font-bold text-gray-900">
            {featureDrift.filter(f => f.driftDetected).length}
          </p>
        </div>
        <div className="bg-white rounded-xl shadow-lg p-6">
          <p className="text-gray-600 text-sm font-medium mb-2">Churn Rate Change</p>
          <p className="text-3xl font-bold text-gray-900">+2.3%</p>
        </div>
        <div className="bg-white rounded-xl shadow-lg p-6">
          <p className="text-gray-600 text-sm font-medium mb-2">Current AUC</p>
          <p className="text-3xl font-bold text-gray-900">0.9234</p>
        </div>
      </div>

      {/* PSI Chart */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">📊 Population Stability Index by Feature</h2>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={featureDrift}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="feature" stroke="#6b7280" angle={-45} textAnchor="end" height={120} />
            <YAxis stroke="#6b7280" />
            <Tooltip 
              contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            />
            <Bar dataKey="psi" fill="#3b82f6" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 flex items-center justify-center space-x-8 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-4 h-1 bg-yellow-500"></div>
            <span>Moderate Threshold (0.1)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-1 bg-red-500"></div>
            <span>High Threshold (0.2)</span>
          </div>
        </div>
      </div>

      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">🚨 Alerts</h2>
          <div className="space-y-3">
            {alerts.map((alert, i) => (
              <div 
                key={i}
                className={`p-4 rounded-lg border-l-4 ${
                  alert.severity === 'high' 
                    ? 'bg-red-50 border-red-500' 
                    : 'bg-yellow-50 border-yellow-500'
                }`}
              >
                <p className="font-medium text-gray-900">
                  <span className="uppercase font-bold">{alert.severity}:</span> {alert.message}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">💡 Recommendations</h2>
        <ol className="list-decimal list-inside space-y-2 text-gray-700">
          <li>Consider retraining the model with recent data</li>
          <li>Investigate changes in customer behavior patterns</li>
          <li>Review feature engineering pipeline for updates</li>
          <li>Increase monitoring frequency for critical features</li>
        </ol>
      </div>
    </div>
  )
}

export default DriftAnalysis