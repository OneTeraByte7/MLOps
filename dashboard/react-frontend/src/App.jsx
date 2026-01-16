import { useState, useEffect } from 'react'
import './App.css'
import Sidebar from './components/Sidebar'
import Overview from './components/Overview'
import ModelPerformance from './components/ModelPerformance'
import DriftAnalysis from './components/DriftAnalysis'
import Predictions from './components/Predictions'
import Explainability from './components/Explainability'

function App() {
  const [currentPage, setCurrentPage] = useState('overview')
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [lastUpdated, setLastUpdated] = useState(new Date())

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        setLastUpdated(new Date())
      }, 30000)
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const renderPage = () => {
    switch (currentPage) {
      case 'overview':
        return <Overview lastUpdated={lastUpdated} />
      case 'performance':
        return <ModelPerformance lastUpdated={lastUpdated} />
      case 'drift':
        return <DriftAnalysis lastUpdated={lastUpdated} />
      case 'predictions':
        return <Predictions lastUpdated={lastUpdated} />
      case 'explainability':
        return <Explainability lastUpdated={lastUpdated} />
      default:
        return <Overview lastUpdated={lastUpdated} />
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar 
        currentPage={currentPage}
        setCurrentPage={setCurrentPage}
        autoRefresh={autoRefresh}
        setAutoRefresh={setAutoRefresh}
        lastUpdated={lastUpdated}
      />
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {renderPage()}
        </div>
      </main>
    </div>
  )
}

export default App