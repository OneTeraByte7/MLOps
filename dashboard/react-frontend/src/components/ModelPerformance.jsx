import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import MetricCard from './MetricCard'
import { Target, TrendingUp, Zap, Award } from 'lucide-react'

const ModelPerformance = ({ lastUpdated }) => {
  const [metrics, setMetrics] = useState({
    auc: 0,
    f1: 0,
    precision: 0,
    recall: 0
  })
  const [performanceData, setPerformanceData] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let isMounted = true
    const controller = new AbortController()
    
    const fetchMLflowData = async () => {
      try {
        setLoading(true)
        const response = await fetch('http://localhost:8000/api/mlflow/runs', {
          signal: controller.signal
        })
        
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
        
        const data = await response.json()
        if (!isMounted) return
        
        const runs = data.runs || []
        
        if (runs.length > 0) {
          setMetrics({
            auc: runs[0].metrics?.test_auc || 0,
            f1: runs[0].metrics?.test_f1 || 0,
            precision: runs[0].metrics?.test_precision || 0,
            recall: runs[0].metrics?.test_recall || 0
          })

          const performanceRuns = runs.map((run, i) => ({
            run: `Run ${i + 1}`,
            auc: run.metrics?.test_auc || 0,
            f1: run.metrics?.test_f1 || 0,
            precision: run.metrics?.test_precision || 0,
            recall: run.metrics?.test_recall || 0
          }))
          setPerformanceData(performanceRuns)
        }
      } catch (error) {
        if (error.name !== 'AbortError') {
          console.error('Failed to fetch MLflow data:', error)
        }
      } finally {
        if (isMounted) {
          setLoading(false)
        }
      }
    }

    fetchMLflowData()
    
    return () => {
      isMounted = false
      controller.abort()
    }
  }, [lastUpdated])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-xl text-gray-600">Loading model performance data...</div>
      </div>
    )
  }

  // Show message if no MLflow data
  if (performanceData.length === 0) {
    return (
      <div className="space-y-6 animate-fade-in">
        <h1 className="text-3xl font-bold text-gray-900">
          🎯 Model Performance Tracking
        </h1>
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-8 text-center">
          <p className="text-xl text-gray-700 mb-4">No MLflow experiment data available.</p>
          <p className="text-gray-600">Train models to see performance metrics here.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-3xl font-bold text-gray-900">
        🎯 Model Performance Tracking
      </h1>

      {/* Latest Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="AUC"
          value={metrics.auc.toFixed(4)}
          icon={Target}
          color="blue"
        />
        <MetricCard
          title="F1 Score"
          value={metrics.f1.toFixed(4)}
          icon={Award}
          color="green"
        />
        <MetricCard
          title="Precision"
          value={metrics.precision.toFixed(4)}
          icon={Zap}
          color="purple"
        />
        <MetricCard
          title="Recall"
          value={metrics.recall.toFixed(4)}
          icon={TrendingUp}
          color="yellow"
        />
      </div>

      {/* Performance Trends */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">📊 Performance Metrics Over Time</h2>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={performanceData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="run" stroke="#6b7280" />
            <YAxis stroke="#6b7280" domain={[0.7, 1.0]} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            />
            <Legend />
            <Line type="monotone" dataKey="auc" stroke="#3b82f6" strokeWidth={2} name="AUC" />
            <Line type="monotone" dataKey="f1" stroke="#22c55e" strokeWidth={2} name="F1" />
            <Line type="monotone" dataKey="precision" stroke="#a855f7" strokeWidth={2} name="Precision" />
            <Line type="monotone" dataKey="recall" stroke="#f59e0b" strokeWidth={2} name="Recall" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Model Comparison Table */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">🔄 Recent Model Runs</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Run</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">AUC</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">F1</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Precision</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Recall</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {performanceData.slice(0, 10).map((row, i) => (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{row.run}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{row.auc.toFixed(4)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{row.f1.toFixed(4)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{row.precision.toFixed(4)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{row.recall.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default ModelPerformance