import React, { useEffect, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

const FraudNetworkGraph = ({ targetNodeId }) => {
  const graphRef = useRef();
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });

  useEffect(() => {
    // In a real scenario, this data comes from your FastAPI backend 
    // returning the specific edges for the detected cluster.
    // Here we generate a realistic mock topology around your targeted Mule account.
    
    const nodes = [];
    const links = [];
    
    // Add the central suspected Mule (C439737079)
    nodes.push({ id: targetNodeId, group: 'mule', val: 10 });

    // Generate a web of surrounding known fraudsters sending money to the mule
    for (let i = 1; i <= 40; i++) {
      const fraudsterId = `Fraud_${i}`;
      nodes.push({ id: fraudsterId, group: 'fraud', val: 3 });
      
      // Edge from fraudster to mule
      links.push({ source: fraudsterId, target: targetNodeId });
      
      // Connect fraudsters to each other to show the "Ring"
      if (i > 1 && Math.random() > 0.5) {
        links.push({ source: fraudsterId, target: `Fraud_${i - 1}` });
      }
    }

    // Add some innocent victim nodes that the fraudsters stole from
    for (let i = 1; i <= 20; i++) {
      const victimId = `Victim_${i}`;
      nodes.push({ id: victimId, group: 'normal', val: 2 });
      links.push({ source: victimId, target: `Fraud_${Math.ceil(Math.random() * 40)}` });
    }

    setGraphData({ nodes, links });
  }, [targetNodeId]);

  return (
    <div className="rounded-xl overflow-hidden border border-white/10 bg-black/50 backdrop-blur-md h-[400px] w-full relative">
      <div className="absolute top-4 left-4 z-10 flex gap-4 text-xs">
        <span className="flex items-center gap-1 text-red-400"><div className="w-3 h-3 rounded-full bg-red-500"></div> Known Fraudsters</span>
        <span className="flex items-center gap-1 text-cyan-400"><div className="w-3 h-3 rounded-full bg-cyan-400"></div> Suspected Mule</span>
        <span className="flex items-center gap-1 text-gray-400"><div className="w-3 h-3 rounded-full bg-gray-500"></div> Normal Accounts</span>
      </div>
      
      <ForceGraph2D
        ref={graphRef}
        graphData={graphData}
        width={800} // Adjust based on your container
        height={400}
        nodeLabel="id"
        nodeColor={node => {
          if (node.group === 'mule') return '#22d3ee'; // Cyan
          if (node.group === 'fraud') return '#ef4444'; // Red
          return '#6b7280'; // Gray
        }}
        nodeRelSize={4}
        linkColor={() => 'rgba(255,255,255,0.1)'}
        linkDirectionalParticles={2} // Creates animated dots flowing along the edges!
        linkDirectionalParticleSpeed={d => 0.01}
        backgroundColor="rgba(0,0,0,0)"
        onEngineStop={() => graphRef.current.zoomToFit(400, 50)}
      />
    </div>
  );
};

export default FraudNetworkGraph;