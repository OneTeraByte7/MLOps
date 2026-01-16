import { useState, useEffect } from 'react'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart } from 'recharts'

const Predictions = ({ lastUpdated }) => {
  const [timeRange, setTimeRange] = useState('24h')
  const [metrics, setMetrics] = useState({
    total: 0,
    churn: 0,
    avgProbability: 0
  })
  const [trendData, setTrendData] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const response = await fetch('http://localhost:8000/api/dashboard/overview')
        const data = await response.json()
        
        const predictions = data.predictions || []
        
        // Calculate metrics
        const churnCount = predictions.filter(p => p.prediction === 'Yes').length
        setMetrics({
          total: predictions.length,
          churn: churnCount,
          avgProbability: data.avg_confidence || 0
        })

        // Trend data by hour
        const hourlyMap = {}
        predictions.forEach(p => {
          const hour = new Date(p.timestamp).getHours()
          if (!hourlyMap[hour]) {
            hourlyMap[hour] = { churnProb: [], churnCount: 0 }
          }
          hourlyMap[hour].churnProb.push(p.churn_probability)
          if (p.prediction === 'Yes') hourlyMap[hour].churnCount++
        })

        const trends = Array.from({ length: 24 }, (_, i) => ({
          time: `${i}:00`,
          churnProb: hourlyMap[i] ? 
            hourlyMap[i].churnProb.reduce((a, b) => a + b, 0) / hourlyMap[i].churnProb.length : 0,
          churnCount: hourlyMap[i] ? hourlyMap[i].churnCount : 0
        }))
        setTrendData(trends)

      } catch (error) {
        console.error('Failed to fetch predictions data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [lastUpdated, timeRange])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-xl text-gray-600">Loading predictions data...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">
          🔮 Prediction Analytics
        </h1>
        <select 
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value)}
          className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          <option value="1h">Last Hour</option>
          <option value="6h">Last 6 Hours</option>
          <option value="24h">Last 24 Hours</option>
          <option value="7d">Last Week</option>
        </select>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <p className="text-gray-600 text-sm font-medium mb-2">Total Predictions</p>
          <p className="text-3xl font-bold text-gray-900">{metrics.total.toLocaleString()}</p>
        </div>
        <div className="bg-white rounded-xl shadow-lg p-6">
          <p className="text-gray-600 text-sm font-medium mb-2">Churn Predictions</p>
          <p className="text-3xl font-bold text-gray-900">{metrics.churn.toLocaleString()}</p>
          <p className="text-sm text-gray-500 mt-1">
            {((metrics.churn / metrics.total) * 100).toFixed(1)}% of total
          </p>
        </div>
        <div className="bg-white rounded-xl shadow-lg p-6">
          <p className="text-gray-600 text-sm font-medium mb-2">Avg Churn Probability</p>
          <p className="text-3xl font-bold text-gray-900">{(metrics.avgProbability * 100).toFixed(1)}%</p>
        </div>
      </div>

      {/* Trend Chart */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">📈 Prediction Trends</h2>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="time" stroke="#6b7280" />
            <YAxis yAxisId="left" stroke="#6b7280" />
            <YAxis yAxisId="right" orientation="right" stroke="#6b7280" />
            <Tooltip 
              contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            />
            <Legend />
            <Line 
              yAxisId="left"
              type="monotone" 
              dataKey="churnProb" 
              stroke="#3b82f6" 
              strokeWidth={3}
              name="Avg Churn Probability"
            />
            <Bar 
              yAxisId="right"
              dataKey="churnCount" 
              fill="#ff7f0e" 
              opacity={0.6}
              name="Churn Count"
              radius={[8, 8, 0, 0]}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* A/B Test Performance */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">🔄 A/B Test Performance</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model Version</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Predictions</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Avg Probability</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Churn Count</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              <tr className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">v1.0.0</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">925</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">0.228</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">211</td>
              </tr>
              <tr className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">v1.1.0</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">617</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">0.245</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">151</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default Predictions