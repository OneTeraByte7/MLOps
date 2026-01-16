import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Info } from 'lucide-react'

const Explainability = ({ lastUpdated }) => {
  const [importanceData, setImportanceData] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const response = await fetch('http://localhost:8000/api/explainability/importance')
        const data = await response.json()
        
        setImportanceData(data.features || [])
      } catch (error) {
        console.error('Failed to fetch explainability data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [lastUpdated])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-xl text-gray-600">Loading explainability data...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-3xl font-bold text-gray-900">
        🔍 Model Explainability
      </h1>

      {/* Global Feature Importance */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">🎯 Global Feature Importance</h2>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={importanceData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis type="number" stroke="#6b7280" />
            <YAxis dataKey="feature" type="category" stroke="#6b7280" width={150} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            />
            <Bar dataKey="importance" fill="#3b82f6" radius={[0, 8, 8, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Feature Rankings */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">📊 Feature Importance Rankings</h2>
        <div className="space-y-3">
          {importanceData.map((item, i) => (
            <div key={i} className="flex items-center space-x-4">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-bold text-sm">
                {i + 1}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-gray-900">{item.feature}</span>
                  <span className="text-sm text-gray-600">{(item.importance * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                    style={{ width: `${item.importance * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Individual Customer Explanation */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">🔬 Individual Customer Explanation</h2>
        <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-4">
          <div className="flex">
            <Info className="text-blue-500 mr-3" size={20} />
            <p className="text-sm text-blue-700">
              Use the <code className="bg-blue-100 px-2 py-1 rounded">/explain</code> API endpoint to get detailed explanations for specific customers
            </p>
          </div>
        </div>
        
        <div className="bg-gray-50 p-4 rounded-lg">
          <p className="text-sm font-medium text-gray-700 mb-2">Example API Call:</p>
          <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
{`curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_001",
    "account_age_days": 365,
    "subscription_tier": "Professional",
    "monthly_revenue": 199.99,
    "logins_per_month": 25,
    "feature_usage_depth": 0.65,
    "support_tickets": 2,
    "avg_ticket_resolution_days": 3.5,
    "nps_score": 8,
    "payment_delays": 0,
    "contract_length_months": 12,
    "team_size": 10,
    "api_calls_per_month": 15000,
    "days_since_last_login": 2
  }'`}
          </pre>
        </div>
      </div>
    </div>
  )
}

export default Explainability