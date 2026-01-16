import { Activity, BarChart3, TrendingUp, Zap, Eye, Settings } from 'lucide-react'

const Sidebar = ({ currentPage, setCurrentPage, autoRefresh, setAutoRefresh, lastUpdated }) => {
  const menuItems = [
    { id: 'overview', icon: Activity, label: 'Overview' },
    { id: 'performance', icon: BarChart3, label: 'Model Performance' },
    { id: 'drift', icon: TrendingUp, label: 'Drift Analysis' },
    { id: 'predictions', icon: Zap, label: 'Predictions' },
    { id: 'explainability', icon: Eye, label: 'Explainability' },
  ]

  return (
    <div className="w-64 bg-gradient-to-b from-indigo-900 to-purple-900 text-white flex flex-col">
      <div className="p-6 border-b border-indigo-700">
        <h1 className="text-2xl font-bold mb-1">MLOps Dashboard</h1>
        <p className="text-indigo-300 text-sm">Churn Prediction System</p>
      </div>

      <nav className="flex-1 px-4 py-6 space-y-2">
        {menuItems.map((item) => (
          <button
            key={item.id}
            onClick={() => setCurrentPage(item.id)}
            className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all ${
              currentPage === item.id
                ? 'bg-white bg-opacity-20 shadow-lg'
                : 'hover:bg-white hover:bg-opacity-10'
            }`}
          >
            <item.icon size={20} />
            <span className="font-medium">{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="p-4 border-t border-indigo-700 space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm">Auto Refresh</span>
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`w-12 h-6 rounded-full transition-colors ${
              autoRefresh ? 'bg-green-500' : 'bg-gray-600'
            }`}
          >
            <div className={`w-5 h-5 bg-white rounded-full transition-transform ${
              autoRefresh ? 'translate-x-6' : 'translate-x-1'
            }`} />
          </button>
        </div>

        <div className="text-xs space-y-1">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            <span>API: Online</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            <span>MLflow: Connected</span>
          </div>
          <div className="text-indigo-300 mt-2">
            Updated: {lastUpdated.toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Sidebar