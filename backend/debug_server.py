import torch
import torch.nn as nn
import numpy as np
import selfies as sf
import uvicorn
import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, QED

# ==========================================
# 1. HARDCODED CONFIG (No Loader Issues)
# ==========================================
# CHANGE THIS IF NEEDED:
MODEL_PATH = "universal_diffusion_model.pth" 

print(f"🚀 DEBUG SERVER STARTING...")
print(f"   -> Looking for model at: {os.path.abspath(MODEL_PATH)}")

if not os.path.exists(MODEL_PATH):
    print("❌ CRITICAL: Model file not found!")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
vocab_size = checkpoint['vocab_size']
props_cols = checkpoint.get('props', [])
scaler = checkpoint.get('scaler', None)
model_state = checkpoint['model_state']
itos = checkpoint['itos']

print(f"✅ LOADED: {len(props_cols)} columns. Device: {device}")

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class SelfiesDiffusion(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, input_dim):
        super().__init__()
        self.max_len = 100 
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(self.max_len, hidden_dim)
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.phys_mlp = nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(), nn.Linear(512, 512))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=2048, batch_first=True, dropout=0.1, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, t, props):
        x_emb = self.emb(x) + self.pos_emb(torch.arange(0, x.size(1), device=x.device).unsqueeze(0))
        if t.dim() == 1: t = t.unsqueeze(-1)
        cond = self.time_mlp(t.float()).unsqueeze(1) + self.phys_mlp(props).unsqueeze(1)
        h = self.transformer(x_emb + cond)
        return self.fc_out(h)

# Initialize Model
model = SelfiesDiffusion(vocab_size, 512, 8, len(props_cols))
model.load_state_dict(model_state)
model.to(device)
model.eval()
print("✅ MODEL READY (Architecture matches state_dict)")

# ==========================================
# 3. FASTAPI APP (Defined Here)
# ==========================================
app = FastAPI()

class DesignRequest(BaseModel):
    domain: str
    mw: float = 350.0

@app.post("/generate")
async def generate(req: DesignRequest):
    print(f"\n🧪 HIT: /generate for {req.domain}")
    trace = []
    
    # 1. BYPASS SCALER (Use Random Noise to prove model works)
    final_vector = np.random.uniform(-1, 1, (1, len(props_cols)))
    
    # 2. SET FLAG
    domain_key = f"is_{req.domain.lower()}"
    for i, col in enumerate(props_cols):
        if domain_key in col.lower():
            final_vector[0, i] = 5.0 # Strong signal
            trace.append(f"Flag Set: {col}")
            break

    # 3. GENERATE
    best_smiles = "C"
    
    x = torch.randint(0, vocab_size, (1, 100)).to(device)
    props_tensor = torch.tensor(final_vector, dtype=torch.float32).to(device)

    for t in reversed(range(30)):
        time_tensor = torch.tensor([t]).float().to(device)
        with torch.no_grad():
            logits = model(x, time_tensor, props_tensor)
        
        logits = logits / 1.0 # Temp
        probs = torch.softmax(logits, dim=-1)
        
        # 🚨 NAN CHECK
        if torch.isnan(probs).any():
            trace.append("❌ NAN DETECTED!")
            break
            
        pred_tokens = torch.multinomial(probs.view(-1, vocab_size), 1).view(x.shape)
        mask = torch.rand_like(x.float()) > (t / 30)
        x = torch.where(mask, pred_tokens, x) 

    # 4. DECODE
    indices = x[0].cpu().tolist()
    tokens = [itos.get(i, "") for i in indices]
    valid_tokens = [t for t in tokens if t not in ["[nop]", "[MASK]"]]
    raw_selfies = "".join(valid_tokens)
    
    try:
        best_smiles = sf.decoder(raw_selfies)
        trace.append(f"✅ Decoded: {best_smiles}")
    except:
        trace.append("❌ Decoder Failed")

    return {
        "smiles": best_smiles,
        "trace": trace,
        "input_cols": len(props_cols)
    }

if __name__ == "__main__":
    # RUN ON PORT 8001 TO AVOID CONFLICTS
    uvicorn.run(app, host="127.0.0.1", port=8001)