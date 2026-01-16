import { TrendingUp, TrendingDown } from 'lucide-react'

const MetricCard = ({ title, value, delta, icon: Icon, trend = 'neutral', color = 'blue' }) => {
  const colorClasses = {
    blue: 'from-blue-500 to-blue-600',
    green: 'from-green-500 to-green-600',
    red: 'from-red-500 to-red-600',
    yellow: 'from-yellow-500 to-yellow-600',
    purple: 'from-purple-500 to-purple-600',
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-gray-600 text-sm font-medium mb-2">{title}</p>
          <p className="text-3xl font-bold text-gray-900">{value}</p>
          {delta && (
            <div className="flex items-center mt-2 space-x-1">
              {trend === 'up' && <TrendingUp size={16} className="text-green-500" />}
              {trend === 'down' && <TrendingDown size={16} className="text-red-500" />}
              <span className={`text-sm font-medium ${
                trend === 'up' ? 'text-green-600' : trend === 'down' ? 'text-red-600' : 'text-gray-600'
              }`}>
                {delta}
              </span>
            </div>
          )}
        </div>
        {Icon && (
          <div className={`w-12 h-12 bg-gradient-to-br ${colorClasses[color]} rounded-lg flex items-center justify-center`}>
            <Icon className="text-white" size={24} />
          </div>
        )}
      </div>
    </div>
  )
}

export default MetricCard