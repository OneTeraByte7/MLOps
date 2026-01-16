import { useState, useEffect } from 'react'
import { Activity, Users, AlertTriangle, Target } from 'lucide-react'
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import MetricCard from './MetricCard'

const Overview = ({ lastUpdated }) => {
  const [metrics, setMetrics] = useState({
    totalPredictions: 0,
    churnRate: 0,
    highRiskCustomers: 0,
    avgConfidence: 0
  })
  const [volumeData, setVolumeData] = useState([])
  const [riskData, setRiskData] = useState([])
  const [recentPredictions, setRecentPredictions] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        
        // Fetch real data from API
        const response = await fetch('http://localhost:8000/api/dashboard/overview')
        const data = await response.json()
        
        setMetrics({
          totalPredictions: data.total_predictions,
          churnRate: (data.churn_rate * 100).toFixed(1),
          highRiskCustomers: data.high_risk_customers,
          avgConfidence: data.avg_confidence
        })

        // Process volume data (predictions by hour)
        const predictions = data.predictions || []
        const hourlyMap = {}
        predictions.forEach(p => {
          const hour = new Date(p.timestamp).getHours()
          hourlyMap[hour] = (hourlyMap[hour] || 0) + 1
        })

        const hours = Array.from({ length: 24 }, (_, i) => ({
          time: `${i}:00`,
          predictions: hourlyMap[i] || 0
        }))
        setVolumeData(hours)

        // Risk data
        const highRisk = predictions.filter(p => p.risk_level === 'High').length
        const mediumRisk = predictions.filter(p => p.risk_level === 'Medium').length
        const lowRisk = predictions.filter(p => p.risk_level === 'Low').length

        setRiskData([
          { name: 'High', value: highRisk, color: '#ef4444' },
          { name: 'Medium', value: mediumRisk, color: '#f59e0b' },
          { name: 'Low', value: lowRisk, color: '#22c55e' }
        ])

        // Recent high-risk customers
        const highRiskCustomers = predictions
          .filter(p => p.risk_level === 'High')
          .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
          .slice(0, 5)
        setRecentPredictions(highRiskCustomers)

      } catch (error) {
        console.error('Failed to fetch overview data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [lastUpdated])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-xl text-gray-600">Loading dashboard data...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">
          🎯 Churn Prediction System Overview
        </h1>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Predictions (24h)"
          value={metrics.totalPredictions.toLocaleString()}
          delta="+12%"
          trend="up"
          icon={Activity}
          color="blue"
        />
        <MetricCard
          title="Predicted Churn Rate"
          value={`${metrics.churnRate}%`}
          delta="-2.3%"
          trend="down"
          icon={Target}
          color="green"
        />
        <MetricCard
          title="High Risk Customers"
          value={metrics.highRiskCustomers}
          delta="+5"
          trend="up"
          icon={AlertTriangle}
          color="red"
        />
        <MetricCard
          title="Avg Confidence"
          value={`${(metrics.avgConfidence * 100).toFixed(0)}%`}
          delta="+1.2%"
          trend="up"
          icon={Users}
          color="purple"
        />
      </div>

      {/* Prediction Volume Chart */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">📈 Prediction Volume</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={volumeData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="time" stroke="#6b7280" />
            <YAxis stroke="#6b7280" />
            <Tooltip 
              contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            />
            <Line 
              type="monotone" 
              dataKey="predictions" 
              stroke="#3b82f6" 
              strokeWidth={3}
              dot={{ fill: '#3b82f6', r: 4 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Risk Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">🎯 Risk Level Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={5}
                dataKey="value"
              >
                {riskData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">📊 Risk Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={riskData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="name" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
              />
              <Bar dataKey="value" fill="#3b82f6" radius={[8, 8, 0, 0]}>
                {riskData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent High-Risk Customers */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">⚠️ Recent High-Risk Customers</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Customer ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Timestamp
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Churn Probability
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Model Version
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {recentPredictions.map((pred, i) => (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {pred.customer_id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {new Date(pred.timestamp).toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800">
                      {(pred.churn_probability * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {pred.model_version}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default Overview