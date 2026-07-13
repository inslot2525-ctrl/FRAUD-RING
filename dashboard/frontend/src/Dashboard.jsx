import { useState, useEffect } from 'react'
import { Shield, Users, AlertTriangle, Search, Loader2 } from 'lucide-react'

const API = 'http://localhost:8000'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function fmt(n) {
  return typeof n === 'number' ? n.toLocaleString() : n
}

function riskColor(level) {
  const map = {
    'CONFIRMED FRAUD': 'text-red-400 border-red-500/50 bg-red-500/10',
    'CRITICAL RISK':   'text-red-400 border-red-500/30 bg-red-500/10',
    'HIGH RISK':       'text-orange-400 border-orange-500/30 bg-orange-500/10',
    'MEDIUM RISK':     'text-yellow-400 border-yellow-500/30 bg-yellow-500/10',
    'LOW RISK':        'text-green-400 border-green-500/30 bg-green-500/10',
  }
  return map[level] ?? 'text-gray-400 border-white/20 bg-white/5'
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------
function MetricCard({ title, value, icon, color = 'text-white', loading }) {
  return (
    <div className="bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-6 flex items-start justify-between">
      <div>
        <p className="text-gray-400 text-sm mb-1">{title}</p>
        {loading
          ? <Loader2 className="animate-spin text-gray-500 mt-2" size={24} />
          : <h3 className={`text-3xl font-light ${color}`}>{fmt(value)}</h3>
        }
      </div>
      <div className={`p-3 bg-white/5 rounded-lg ${color}`}>{icon}</div>
    </div>
  )
}

function RingsTable({ rings, loading }) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-40 gap-2 text-gray-400">
        <Loader2 className="animate-spin" size={20} /> Loading rings...
      </div>
    )
  }
  if (!rings.length) return <p className="text-gray-500 text-sm">No rings data available.</p>

  return (
    <table className="w-full text-left border-collapse">
      <thead>
        <tr className="border-b border-white/10 text-gray-400 text-sm">
          <th className="pb-3">Ring</th>
          <th className="pb-3">Total Accounts</th>
          <th className="pb-3">Known Fraudsters</th>
          <th className="pb-3 text-cyan-400">Suspected Mules</th>
          <th className="pb-3 text-gray-400">Top Target</th>
        </tr>
      </thead>
      <tbody className="text-sm">
        {rings.map((r) => (
          <tr
            key={r.cluster_id}
            className={`border-b border-white/5 ${r.rank === 1 ? 'bg-red-500/10' : ''}`}
          >
            <td className="py-3">Ring #{r.cluster_id}</td>
            <td>{fmt(r.total_accounts)}</td>
            <td>{fmt(r.known_fraudsters)}</td>
            <td className="font-bold text-cyan-400">{fmt(r.suspected_mules)}</td>
            <td className="text-gray-400 font-mono text-xs">{r.top_targets?.[0] ?? '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// ---------------------------------------------------------------------------
// Main Dashboard
// ---------------------------------------------------------------------------
export default function Dashboard() {
  const [stats,         setStats]         = useState(null)
  const [rings,         setRings]         = useState([])
  const [statsLoading,  setStatsLoading]  = useState(true)
  const [ringsLoading,  setRingsLoading]  = useState(true)
  const [apiError,      setApiError]      = useState(null)

  const [searchQuery,   setSearchQuery]   = useState('')
  const [investigation, setInvestigation] = useState(null)
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError,   setSearchError]   = useState(null)

  // Fetch stats and rings on mount
  useEffect(() => {
    fetch(`${API}/api/stats`)
      .then(r => r.json())
      .then(d => { setStats(d); setStatsLoading(false) })
      .catch(() => { setApiError('Cannot reach API — is the server running?'); setStatsLoading(false) })

    fetch(`${API}/api/rings?top_n=10`)
      .then(r => r.json())
      .then(d => { setRings(d.rings ?? []); setRingsLoading(false) })
      .catch(() => setRingsLoading(false))
  }, [])

  // Investigate account
  async function handleSearch(e) {
    e.preventDefault()
    if (!searchQuery.trim()) return
    setSearchLoading(true)
    setInvestigation(null)
    setSearchError(null)
    try {
      const res = await fetch(`${API}/api/investigate/${searchQuery.trim()}`)
      if (!res.ok) {
        const err = await res.json()
        setSearchError(err.detail ?? 'Account not found.')
      } else {
        setInvestigation(await res.json())
      }
    } catch {
      setSearchError('Cannot reach API.')
    } finally {
      setSearchLoading(false)
    }
  }

  return (
    <div className="relative min-h-screen text-white font-sans bg-black">
      <div className="relative z-10 p-8 min-h-screen bg-gradient-to-br from-black via-gray-950 to-black">

        {/* API error banner */}
        {apiError && (
          <div className="mb-6 p-4 rounded-xl border border-yellow-500/40 bg-yellow-500/10 text-yellow-300 text-sm">
            ⚠ {apiError}
          </div>
        )}

        {/* Header */}
        <header className="flex justify-between items-center mb-10 border-b border-white/10 pb-4">
          <div className="flex items-center gap-3">
            <Shield className="text-cyan-400 w-8 h-8" />
            <h1 className="text-3xl font-light tracking-wider">
              FRAUD<span className="font-bold">RING</span> DETECTOR
            </h1>
          </div>
          <span className="text-xs text-gray-500 border border-white/10 px-4 py-2 rounded-full">
            Live · GNN Engine
          </span>
        </header>

        {/* Metric cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10">
          <MetricCard title="Accounts Scanned"    value={stats?.total_nodes}      icon={<Users size={20}/>}         loading={statsLoading} />
          <MetricCard title="Total Transactions"  value={stats?.total_edges}      icon={<AlertTriangle size={20}/>} loading={statsLoading} />
          <MetricCard title="Known Fraudsters"    value={stats?.known_fraudsters} icon={<Shield size={20}/>}        loading={statsLoading} color="text-red-400" />
          <MetricCard title="Suspected Mules"     value={stats?.suspected_mules}  icon={<Search size={20}/>}        loading={statsLoading} color="text-cyan-400" />
        </div>

        {/* Main grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Rings table */}
          <div className="col-span-2 bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-6">
            <h2 className="text-xl font-medium mb-4">Critical Threat Networks</h2>
            <RingsTable rings={rings} loading={ringsLoading} />
          </div>

          {/* Investigate */}
          <div className="col-span-1 bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-6">
            <h2 className="text-xl font-medium mb-4">Investigate Node</h2>
            <form onSubmit={handleSearch} className="mb-4">
              <input
                type="text"
                placeholder="Enter Account ID (e.g. C439737079)"
                className="w-full bg-black/50 border border-white/20 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-cyan-400 text-sm mb-3"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
              />
              <button
                type="submit"
                disabled={searchLoading}
                className="w-full bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-200 py-2 rounded-lg transition-all text-sm flex items-center justify-center gap-2"
              >
                {searchLoading ? <><Loader2 size={14} className="animate-spin"/> Analysing...</> : 'Investigate'}
              </button>
            </form>

            {searchError && (
              <p className="text-yellow-400 text-sm p-3 border border-yellow-500/30 bg-yellow-500/10 rounded-lg">
                {searchError}
              </p>
            )}

            {investigation && (
              <div className={`p-4 border rounded-lg ${riskColor(investigation.risk_level)}`}>
                <h3 className="font-bold flex items-center gap-2 mb-2">
                  <AlertTriangle size={16} /> {investigation.risk_level}
                </h3>
                <p className="text-xs text-gray-300 mb-3">{investigation.description}</p>
                <div className="text-xs space-y-1 text-gray-400">
                  <p>Cluster ID    : <span className="text-white">{investigation.cluster_id}</span></p>
                  <p>Cluster size  : <span className="text-white">{fmt(investigation.cluster_size)}</span></p>
                  <p>Fraud density : <span className="text-white">{(investigation.fraud_ratio * 100).toFixed(1)}%</span></p>
                </div>
                {investigation.risk_level !== 'LOW RISK' && (
                  <button className="mt-4 w-full bg-red-500/20 hover:bg-red-500/40 border border-red-500/50 text-red-200 py-2 rounded transition-all text-sm">
                    Flag Account for Review
                  </button>
                )}
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  )
}
