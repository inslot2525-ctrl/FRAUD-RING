import React, { useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

// We now pass graphData directly as a prop from the backend!
const FraudNetworkGraph = ({ graphData }) => {
  const graphRef = useRef();

  if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
    return <div className="text-white/50 p-4 text-center">No topology data available from backend.</div>;
  }

  return (
    <div className="rounded-xl overflow-hidden border border-white/10 bg-black/50 backdrop-blur-md h-[400px] w-full relative">
      <div className="absolute top-4 left-4 z-10 flex gap-4 text-xs">
        <span className="flex items-center gap-1 text-red-400"><div className="w-3 h-3 rounded-full bg-red-500"></div> Known Fraudsters</span>
        <span className="flex items-center gap-1 text-cyan-400"><div className="w-3 h-3 rounded-full bg-cyan-400"></div> Suspected Mule</span>
        <span className="flex items-center gap-1 text-gray-400"><div className="w-3 h-3 rounded-full bg-gray-500"></div> Normal Accounts</span>
      </div>
      
      <ForceGraph2D
        ref={graphRef}
        graphData={graphData} // Fed directly from FastAPI
        width={800} 
        height={400}
        nodeLabel="id"
        nodeColor={node => {
          if (node.group === 'mule') return '#22d3ee'; // Cyan
          if (node.group === 'fraud') return '#ef4444'; // Red
          return '#6b7280'; // Gray
        }}
        nodeRelSize={5}
        linkColor={() => 'rgba(255,255,255,0.15)'}
        linkDirectionalParticles={2} 
        linkDirectionalParticleSpeed={0.008}
        backgroundColor="rgba(0,0,0,0)"
        onEngineStop={() => graphRef.current?.zoomToFit(400, 50)}
      />
    </div>
  );
};

export default FraudNetworkGraph;