import pandas as pd
import numpy as np
import random

print("⚙️ Booting up Synthetic Fraud-Ring Generator...")

# Configuration
NUM_NORMAL_USERS = 5000
NUM_MERCHANTS = 500
NUM_NORMAL_TX = 48000
NUM_FRAUD_RINGS = 3

# Generate User IDs
normal_users = [f"U_{i}" for i in range(NUM_NORMAL_USERS)]
merchants = [f"M_{i}" for i in range(NUM_MERCHANTS)]

transactions = []

print(f"💸 Generating {NUM_NORMAL_TX} normal background transactions...")
# 1. GENERATE BACKGROUND NOISE (Normal Behavior)
for _ in range(NUM_NORMAL_TX):
    sender = random.choice(normal_users)
    # Users usually send to merchants, sometimes to other users
    receiver = random.choice(merchants) if random.random() > 0.2 else random.choice(normal_users)
    amount = round(random.uniform(5.0, 1500.0), 2)
    transactions.append({"Sender_ID": sender, "Receiver_Acct": receiver, "Tx_Amount": amount})

print(f"🕷️ Injecting {NUM_FRAUD_RINGS} massive fraud syndicates...")
# 2. INJECT FRAUD RINGS (The Signal for the GNN)
for ring_id in range(NUM_FRAUD_RINGS):
    mule_hub = f"MULE_HUB_CRITICAL_{ring_id}"
    offshore_acct = f"OFFSHORE_CAYMAN_{ring_id}"
    
    # Each ring has 50-100 fraudsters
    num_fraudsters = random.randint(50, 100)
    fraudsters = [f"FRAUD_ACT_{ring_id}_{i}" for i in range(num_fraudsters)]
    
    # Each fraudster steals from 5-15 victims
    for f_node in fraudsters:
        num_victims = random.randint(5, 15)
        for _ in range(num_victims):
            victim = random.choice(normal_users)
            amount = round(random.uniform(500.0, 3000.0), 2)
            # Victim -> Fraudster
            transactions.append({"Sender_ID": victim, "Receiver_Acct": f_node, "Tx_Amount": amount})
        
        # Fraudster -> Mule Hub (Consolidating the stolen funds)
        transactions.append({"Sender_ID": f_node, "Receiver_Acct": mule_hub, "Tx_Amount": round(random.uniform(5000.0, 20000.0), 2)})
    
    # Mule Hub -> Offshore (Cash out)
    transactions.append({"Sender_ID": mule_hub, "Receiver_Acct": offshore_acct, "Tx_Amount": round(random.uniform(100000.0, 500000.0), 2)})

# 3. SHUFFLE AND SAVE
print("🔀 Shuffling ledger to hide the network traces...")
df = pd.DataFrame(transactions)
df = df.sample(frac=1).reset_index(drop=True) # Shuffle the rows

filename = "large_test_ledger.csv"
df.to_csv(filename, index=False)
print(f"✅ Success! Saved {len(df)} transactions to {filename}.")