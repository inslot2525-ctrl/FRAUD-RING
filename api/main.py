from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze_network(
    file: UploadFile = File(...),
    sender_col: str = Form(...),
    receiver_col: str = Form(...)
):
    print(f"📥 Processing {file.filename} through GNN...")
    
    # Simulate processing time
    await asyncio.sleep(2.5)

    # Return the exact JSON payload, now including the "graph_data" topology
    return {
        "status": "success",
        "metrics": {
            "total_nodes": "9,073,900",
            "total_edges": "6,362,620",
            "known_fraudsters": "16,382",
            "suspected_mules": "33,349"
        },
        "clusters": [
            {"id": "Ring #102", "total": 8214, "fraud": 8213, "mules": 1, "type": "Cash-Out Hub", "target": "C439737079"},
            {"id": "Ring #125", "total": 2685, "fraud": 2678, "mules": 7, "type": "Laundering Chain", "target": "M360644783"}
        ],
        "investigation_db": {
            "C439737079": {
                "status": "CRITICAL RISK",
                "cluster": "102",
                "description": "Account sits at the center of 8,213 fraudsters.",
                # THIS IS NEW: The actual topology data for the React Graph
                "graph_data": {
                    "nodes": [
                        {"id": "C439737079", "group": "mule"},
                        {"id": "F_8831", "group": "fraud"},
                        {"id": "F_9921", "group": "fraud"},
                        {"id": "F_1023", "group": "fraud"},
                        {"id": "F_4421", "group": "fraud"},
                        {"id": "F_5532", "group": "fraud"},
                        {"id": "Victim_A", "group": "normal"},
                        {"id": "Victim_B", "group": "normal"}
                    ],
                    "links": [
                        {"source": "F_8831", "target": "C439737079"},
                        {"source": "F_9921", "target": "C439737079"},
                        {"source": "F_1023", "target": "C439737079"},
                        {"source": "F_4421", "target": "C439737079"},
                        {"source": "F_5532", "target": "C439737079"},
                        {"source": "F_8831", "target": "F_9921"},
                        {"source": "Victim_A", "target": "F_8831"},
                        {"source": "Victim_B", "target": "F_1023"}
                    ]
                }
            }
        }
    }