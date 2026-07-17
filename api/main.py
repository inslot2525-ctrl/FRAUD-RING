from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import asyncio
import io
import os

app = FastAPI(title="FraudGNN Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")
        
    print(f"📥 Received file: {file.filename} -> Initiating Smart Auto-Mapper...")
    
    # 1. Safely read the uploaded file into a Pandas DataFrame
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse CSV file.")

    # 2. SMART AUTO-MAPPER DICTIONARY
    mapping_rules = {
        'nameOrig': ['sender', 'origin', 'source', 'nameorig', 'client_id', 'customer_id', 'sender_id'],
        'nameDest': ['receiver', 'destination', 'target', 'namedest', 'merchant_id', 'beneficiary', 'receiver_acct'],
        'amount': ['amount', 'tx_amount', 'value', 'transaction_amount', 'usd']
    }

    df_cols_lower = {col.lower(): col for col in df.columns}
    rename_dict = {}

    for target_col, synonyms in mapping_rules.items():
        if target_col in df.columns:
            continue 
            
        found = False
        for syn in synonyms:
            if syn in df_cols_lower:
                rename_dict[df_cols_lower[syn]] = target_col
                found = True
                break
                
        if not found:
            raise HTTPException(
                status_code=400, 
                detail=f"Auto-Mapper Failed: Could not find a column for '{target_col}'."
            )

    df = df.rename(columns=rename_dict)

    if 'isFraud' not in df.columns:
        df['isFraud'] = 0

    # Save cleaned data to disk for PyTorch
    cleaned_filepath = "temp_cleaned_ledger.csv"
    df.to_csv(cleaned_filepath, index=False)
    
    # =================================================================
    # PYTORCH EXECUTION BLOCK (Call your actual models here)
    # build_pyg_graph(filepath=cleaned_filepath)
    # =================================================================
    
    await asyncio.sleep(1.5) # Simulate processing time

    # =================================================================
    # DYNAMIC GRAPH GENERATION FROM UPLOADED DATA
    # =================================================================
    
    # Cap visualization at 800 edges to prevent browser memory crashes
    vis_df = df.head(800)
    
    # Extract unique nodes from the data
    senders = vis_df['nameOrig'].dropna().unique().tolist()
    receivers = vis_df['nameDest'].dropna().unique().tolist()
    unique_nodes = list(set(senders + receivers))

    nodes_list = []
    fraud_count = 0
    mule_count = 0

    for node in unique_nodes:
        node_str = str(node).upper()
        group = "normal"
        
        # Color code based on the generated data labels
        if "FRAUD" in node_str:
            group = "fraud"
            fraud_count += 1
        elif "MULE" in node_str or "OFFSHORE" in node_str:
            group = "mule"
            mule_count += 1
            
        nodes_list.append({"id": str(node), "group": group})

    # Extract dynamic links from the data
    links_list = []
    for _, row in vis_df.iterrows():
        links_list.append({
            "source": str(row['nameOrig']),
            "target": str(row['nameDest'])
        })

    # Clean up the temp file
    if os.path.exists(cleaned_filepath):
        os.remove(cleaned_filepath)
    
    return {
        "status": "success",
        "metrics": {
            "total_nodes": len(pd.concat([df['nameOrig'], df['nameDest']]).unique()),
            "total_edges": len(df),
            "known_fraudsters": fraud_count,
            "suspected_mules": mule_count
        },
        "graph_data": {
            "nodes": nodes_list,
            "links": links_list
        }
    }