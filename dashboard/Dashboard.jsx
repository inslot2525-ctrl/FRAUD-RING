import React, { useState, useRef } from 'react';
import AnoAI from './AnoAI'; // Your 3D meteor background component
import { Shield, Users, AlertTriangle, Search, Upload, Loader2 } from 'lucide-react';

const Dashboard = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [dashboardData, setDashboardData] = useState(null); // Holds the FastAPI response
  
  // File upload & column mapping states
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [showMappingModal, setShowMappingModal] = useState(false);
  const [mappings, setMappings] = useState({
    sender: 'nameOrig',
    receiver: 'nameDest',
    amount: 'amount'
  });

  // Handle file selection and pop open the mapping modal
  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setShowMappingModal(true);
    }
  };

  // Execute the fetch request to your PyTorch/FastAPI backend
  const runAnalysis = async () => {
    if (!selectedFile) return;
    
    setIsAnalyzing(true);
    setShowMappingModal(false); // Hide modal while loading

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("sender_col", mappings.sender);
      formData.append("receiver_col", mappings.receiver);
      formData.append("amount_col", mappings.amount);

      // Call the FastAPI Python server!
      const response = await fetch("http://localhost:8000/api/analyze", {
          method: "POST",
          body: formData
      });

      const data = await response.json();
      setDashboardData(data); // Populate UI with actual model results
    } catch (error) {
      console.error("Error analyzing network:", error);
      alert("Failed to connect to the AI Backend. Ensure FastAPI is running on port 8000.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Safe variable fallbacks
  const metrics = dashboardData?.metrics || {};
  const clusters = dashboardData?.clusters || [];
  const investigationDb = dashboardData?.investigation_db || {};
  const searchResult = investigationDb[searchQuery];

  return (
    <div className="relative min-h-screen text-white font-sans bg-black overflow-hidden">
      {/* 1. Background Layer (WebGL Shader) */}
      <div className="absolute inset-0 z-0">
        <AnoAI />
      </div>

      {/* 2. Glassmorphism UI Layer */}
      <div className="relative z-10 p-8 min-h-screen bg-black/40 backdrop-blur-sm overflow-y-auto">
        
        {/* Header */}
        <header className="flex justify-between items-center mb-10 border-b border-white/10 pb-4">
          <div className="flex items-center gap-3">
            <Shield className="text-cyan-400 w-8 h-8" />
            <h1 className="text-3xl font-light tracking-wider">FRAUD<span className="font-bold">RING</span> DETECTOR</h1>
          </div>
          
          <div>
            <input 
              type="file" 
              accept=".csv" 
              className="hidden" 
              ref={fileInputRef} 
              onChange={handleFileSelect} 
            />
            <button 
              onClick={() => fileInputRef.current?.click()}
              disabled={isAnalyzing}
              className="flex items-center gap-2 bg-white/10 hover:bg-white/20 border border-white/20 px-6 py-2 rounded-full transition-all text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isAnalyzing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
              {isAnalyzing ? "Analyzing Graph Topology..." : "Upload CSV Network"}
            </button>
          </div>
        </header>

        {/* Column Mapping Modal */}
        {showMappingModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md">
            <div className="bg-[#0a0a0a] border border-white/20 p-8 rounded-2xl w-full max-w-md shadow-2xl shadow-cyan-900/20">
              <h2 className="text-2xl font-light mb-2">Map CSV Columns</h2>
              <p className="text-sm text-gray-400 mb-6">Align your dataset with the GNN schema.</p>
              
              <div className="space-y-4 mb-6 text-sm text-gray-300">
                <div>
                  <label className="block mb-1">Sender Column Name</label>
                  <input type="text" className="w-full bg-white/5 border border-white/20 p-2.5 rounded focus:outline-none focus:border-cyan-400 text-white" value={mappings.sender} onChange={e => setMappings({...mappings, sender: e.target.value})} />
                </div>
                <div>
                  <label className="block mb-1">Receiver Column Name</label>
                  <input type="text" className="w-full bg-white/5 border border-white/20 p-2.5 rounded focus:outline-none focus:border-cyan-400 text-white" value={mappings.receiver} onChange={e => setMappings({...mappings, receiver: e.target.value})} />
                </div>
                <div>
                  <label className="block mb-1">Amount Column Name</label>
                  <input type="text" className="w-full bg-white/5 border border-white/20 p-2.5 rounded focus:outline-none focus:border-cyan-400 text-white" value={mappings.amount} onChange={e => setMappings({...mappings, amount: e.target.value})} />
                </div>
              </div>
              <div className="flex justify-end gap-3">
                <button onClick={() => setShowMappingModal(false)} className="px-4 py-2 text-gray-400 hover:text-white transition-colors">Cancel</button>
                <button onClick={runAnalysis} className="bg-cyan-500 hover:bg-cyan-400 text-black font-bold px-6 py-2 rounded transition-colors shadow-[0_0_15px_rgba(6,182,212,0.4)]">Initialize Engine</button>
              </div>
            </div>
          </div>
        )}

        {/* Default Idle State (Before Upload) */}
        {!dashboardData && !isAnalyzing && (
            <div className="flex flex-col items-center justify-center mt-32 text-center animate-pulse">
                <Shield className="w-24 h-24 text-white/10 mb-6" />
                <h2 className="text-4xl font-light text-white/40 mb-4">GNN Engine Idle</h2>
                <p className="text-white/30 max-w-lg">Upload a transaction dataset to map the hidden topology of mule networks.</p>
            </div>
        )}

        {/* Dashboard View (After Upload & API Response) */}
        {dashboardData && (
          <div className="animate-fade-in">
            {/* Top Metrics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10">
              <MetricCard title="Accounts Scanned" value={metrics.total_nodes} icon={<Users />} />
              <MetricCard title="Total Transactions" value={metrics.total_edges} icon={<AlertTriangle />} />
              <MetricCard title="Known Fraudsters" value={metrics.known_fraudsters} icon={<Shield />} color="text-red-400" border="border-red-500/30" />
              <MetricCard title="Hidden Mules Found" value={metrics.suspected_mules} icon={<Search />} color="text-cyan-400" border="border-cyan-500/30" />
            </div>

            {/* Main Content Area */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              {/* Left Column: Top Rings */}
              <div className="col-span-2 bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-6">
                <h2 className="text-xl font-medium mb-4">Critical Threat Networks</h2>
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-white/10 text-gray-400 text-sm">
                        <th className="pb-3 font-medium">Cluster ID</th>
                        <th className="pb-3 font-medium">Threat Type</th>
                        <th className="pb-3 font-medium">Total Accounts</th>
                        <th className="pb-3 font-medium">Known Fraud</th>
                        <th className="pb-3 font-medium text-cyan-400">Hidden Mules</th>
                      </tr>
                    </thead>
                    <tbody className="text-sm">
                      {clusters.map((cluster, idx) => (
                        <tr key={idx} className={`border-b border-white/5 transition-colors hover:bg-white/5 ${cluster.mules > 0 && cluster.fraud > 0 ? 'bg-red-500/10' : ''}`}>
                          <td className="py-4 text-gray-300">{cluster.id}</td>
                          <td className="py-4 text-gray-300">{cluster.type}</td>
                          <td className="py-4">{cluster.total.toLocaleString()}</td>
                          <td className={`py-4 ${cluster.fraud > 0 ? 'text-red-400 font-medium' : 'text-gray-500'}`}>{cluster.fraud.toLocaleString()}</td>
                          <td className="py-4 font-bold text-cyan-400">
                            {cluster.mules.toLocaleString()} 
                            {cluster.target !== "N/A" && <span className="text-xs font-normal text-cyan-200/50 block mt-1">Target: {cluster.target}</span>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Right Column: Deep Investigation Search */}
              <div className="col-span-1 bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-6">
                <h2 className="text-xl font-medium mb-4">Investigate Node</h2>
                <div className="relative mb-6">
                  <input 
                    type="text" 
                    placeholder="Enter Account ID (e.g., C439737079)"
                    className="w-full bg-black/50 border border-white/20 rounded-lg py-3 pl-4 pr-10 text-white focus:outline-none focus:border-cyan-400 transition-colors"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                  <Search className="absolute right-3 top-3 text-gray-500 w-5 h-5" />
                </div>
                
                {searchResult ? (
                  <div className="p-5 border border-red-500/30 bg-red-500/10 rounded-xl transition-all shadow-[0_0_20px_rgba(239,68,68,0.15)]">
                    <h3 className="text-red-400 font-bold flex items-center gap-2 mb-3">
                      <AlertTriangle size={18} /> {searchResult.status}
                    </h3>
                    <div className="space-y-2 mb-4">
                      <p className="text-sm text-gray-300 bg-black/30 p-2 rounded">
                        <span className="text-gray-500 mr-2">Cluster:</span> Ring {searchResult.cluster}
                      </p>
                      <p className="text-sm text-gray-300 bg-black/30 p-2 rounded leading-relaxed">
                        {searchResult.description}
                      </p>
                    </div>
                    <button className="w-full bg-red-500/20 hover:bg-red-500/40 border border-red-500/50 text-red-200 py-2.5 rounded transition-all font-medium flex justify-center items-center gap-2">
                      <Shield size={16} /> Freeze Node
                    </button>
                  </div>
                ) : searchQuery.length > 3 ? (
                  <div className="p-6 border border-white/5 bg-black/20 rounded-xl text-center">
                    <Shield className="w-10 h-10 text-emerald-500/30 mx-auto mb-3" />
                    <p className="text-base text-gray-300 font-medium">Account clear</p>
                    <p className="text-sm text-gray-500 mt-1 leading-relaxed">No topological proximity to known fraud clusters detected in the embedding space.</p>
                  </div>
                ) : null}
              </div>

            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Reusable micro-component for metrics
const MetricCard = ({ title, value, icon, color = "text-white", border = "border-white/10" }) => (
  <div className={`bg-white/5 border ${border} backdrop-blur-md rounded-2xl p-6 flex items-start justify-between transition-transform hover:scale-[1.02] duration-300`}>
    <div>
      <p className="text-gray-400 text-sm mb-1">{title}</p>
      <h3 className={`text-3xl font-light tracking-tight ${color}`}>{value}</h3>
    </div>
    <div className={`p-3 bg-white/5 border ${border} rounded-xl ${color}`}>
      {icon}
    </div>
  </div>
);

export default Dashboard;