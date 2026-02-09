import torch
import torch.nn as nn
import numpy as np
import selfies as sf
import uvicorn
import os
import sys
import requests
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any
from contextlib import asynccontextmanager
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, QED

# ==========================================
# 1. CONFIGURATION & GLOBALS
# ==========================================
# Automatically find the model in the same folder as this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "universal_diffusion_model.pth"
MODEL_PATH = os.path.join(CURRENT_DIR, MODEL_FILENAME)

# Global Memory (The "Brain")
ml_context = {
    "model": None,
    "scaler": None,
    "props_cols": [],
    "vocab_size": 0,
    "itos": {},
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ==========================================
# 2. MODEL DEFINITION (Hardcoded here)
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

# ==========================================
# 3. LIFESPAN (Startup Logic)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print(f"🚀 MONOLITH STARTUP: {MODEL_PATH}")
    print(f"   -> Device: {ml_context['device']}")
    print("="*60)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ CRITICAL: Model file not found at {MODEL_PATH}")
        yield
        return

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=ml_context['device'], weights_only=False)
        ml_context['vocab_size'] = checkpoint['vocab_size']
        ml_context['props_cols'] = checkpoint.get('props', [])
        ml_context['scaler'] = checkpoint.get('scaler', None)
        ml_context['itos'] = checkpoint['itos']
        
        # Initialize Model
        input_dim = len(ml_context['props_cols'])
        model = SelfiesDiffusion(ml_context['vocab_size'], 512, 8, input_dim)
        model.load_state_dict(checkpoint['model_state'])
        model.to(ml_context['device'])
        model.eval()
        
        ml_context['model'] = model
        
        # Check for Drug Flag
        drug_idx = next((i for i, c in enumerate(ml_context['props_cols']) if 'drug' in c), -1)
        print(f"✅ LOADED SUCCESSFULLY! (Inputs: {input_dim})")
        print(f"   -> 'is_drug' found at Index: {drug_idx}")
        
    except Exception as e:
        print(f"❌ FATAL LOAD ERROR: {e}")
        traceback.print_exc()
        
    yield
    ml_context.clear()

# ==========================================
# 4. API SETUP
# ==========================================
app = FastAPI(title="ChemGen Monolith", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow EVERYTHING
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DesignRequest(BaseModel):
    domain: str
    mw: Optional[float] = None
    logp: Optional[float] = None
    tpsa: Optional[float] = None
    hbd: Optional[float] = None
    hba: Optional[float] = None
    rot: Optional[float] = None
    rings: Optional[float] = None
    charge: Optional[float] = None
    qed: Optional[float] = None

class SmilesRequest(BaseModel):
    smiles: str

# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================
def run_diffusion_safe(vector, steps=30, temp=1.0):
    logs = []
    try:
        model = ml_context['model']
        device = ml_context['device']
        vocab_size = ml_context['vocab_size']
        
        x = torch.randint(0, vocab_size, (1, 100)).to(device)
        props_tensor = torch.tensor(vector, dtype=torch.float32).to(device)
        
        for t in reversed(range(steps)):
            time_tensor = torch.tensor([t]).float().to(device)
            with torch.no_grad():
                logits = model(x, time_tensor, props_tensor)
            
            logits = logits / temp 
            probs = torch.softmax(logits, dim=-1)
            
            # 🚨 NAN CHECK (Detects broken scaler inputs)
            if torch.isnan(probs).any():
                return None, ["❌ NaN Detected"]
                
            pred_tokens = torch.multinomial(probs.view(-1, vocab_size), 1).view(x.shape)
            mask = torch.rand_like(x.float()) > (t / steps)
            x = torch.where(mask, pred_tokens, x) 
            
        indices = x[0].cpu().tolist()
        tokens = [ml_context['itos'].get(i, "") for i in indices]
        valid = [t for t in tokens if t not in ["[nop]", "[MASK]"]]
        raw_selfies = "".join(valid)
        
        try: 
            return sf.decoder(raw_selfies), ["✅ Decoded"]
        except: 
            return None, ["⚠️ Decoder Failed"]
    except Exception as e:
        return None, [f"❌ Crash: {e}"]

def get_pubchem_id(smiles):
    if not smiles: return None
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(smiles)}/cids/JSON"
        r = requests.get(url, timeout=1.5)
        if r.status_code == 200:
            cid = str(r.json()["IdentifierList"]["CID"][0])
            return cid if cid != "0" else None
    except: return None

# ==========================================
# 6. ENDPOINTS
# ==========================================
@app.post("/generate")
async def generate(req: DesignRequest):
    print(f"🚀 API HIT: /generate for {req.domain}")
    trace = []
    
    if not ml_context['model']:
        return {"error": "Model not loaded", "trace": ["❌ Model Context Empty"]}

    props_cols = ml_context['props_cols']
    scaler = ml_context['scaler']
    
    # --- STRATEGY: Try Scaler, Fallback to Noise ---
    final_vector = np.zeros((1, len(props_cols)))
    use_noise = False
    
    try:
        # Map inputs
        scaler_feats = ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rot', 'rings', 'charge', 'qed', 'adhesion', 'toxicity', 'affinity']
        raw = [getattr(req, f, scaler.mean_[i] if scaler else 0) for i, f in enumerate(scaler_feats)]
        
        if scaler:
            final_vector[0, :12] = scaler.transform([raw])[0]
            trace.append("✅ Scaler Used")
        else:
            use_noise = True
    except:
        trace.append("⚠️ Scaler Failed (Version Mismatch). Using Noise.")
        use_noise = True
        
    if use_noise or np.isnan(final_vector).any():
        final_vector = np.random.uniform(-1, 1, (1, len(props_cols)))
        trace.append("⚡ ACTIVATED FALLBACK NOISE")

    # Set Flag
    domain_key = f"is_{req.domain.lower()}"
    for i, col in enumerate(props_cols):
        if domain_key in col.lower():
            final_vector[0, i] = 5.0
            trace.append(f"✅ Flag Set: {col}")
            break

    # Generation Loop
    best_smiles = "C"
    final_mol = None
    
    for attempt in range(5):
        smiles, logs = run_diffusion_safe(final_vector, steps=30, temp=1.0)
        if logs and "NaN" in logs[0]:
            trace.append("🔄 NaNs detected. Retrying with fresh noise.")
            final_vector = np.random.uniform(-1, 1, (1, len(props_cols)))
            continue
            
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                best_smiles = smiles
                final_mol = mol
                trace.append("🎉 Success!")
                break
    
    # Packaging
    if not final_mol: final_mol = Chem.MolFromSmiles(best_smiles)
    pub_id = get_pubchem_id(best_smiles)
    
    properties = {"valid": False}
    mol_block = ""
    if final_mol:
        try:
            mol_3d = Chem.AddHs(final_mol)
            AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
            mol_block = Chem.MolToMolBlock(mol_3d)
            properties = {
                "mw": round(Descriptors.MolWt(final_mol), 2),
                "logp": round(Descriptors.MolLogP(final_mol), 2),
                "is_novel": (pub_id is None),
                "pubchem_id": pub_id,
                "valid": True
            }
        except: pass

    return {
        "smiles": best_smiles,
        "mol_block": mol_block,
        "properties": properties,
        "domain": req.domain,
        "trace": trace
    }

@app.post("/analyze")
async def analyze(req: SmilesRequest):
    mol = Chem.MolFromSmiles(req.smiles)
    if not mol: return {"error": "Invalid SMILES"}
    return {
        "mw": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rot": Descriptors.NumRotatableBonds(mol),
        "rings": Lipinski.RingCount(mol),
        "qed": round(QED.qed(mol), 3),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)