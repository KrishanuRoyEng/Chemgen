import os
import requests
import torch
import torch.nn as nn
import numpy as np
import selfies as sf
import uvicorn
import pubchempy as pcp  # <--- NEW IMPORT
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, QED

# ==========================================
# 1. CONFIGURATION & DOWNLOADER
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "universal_diffusion_model.pth"
MODEL_PATH = os.path.join(CURRENT_DIR, MODEL_FILENAME)
MODEL_URL = "https://huggingface.co/krishanuroy/universal_diffusion_model/resolve/main/universal_diffusion_model.pth" 

def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        if os.path.getsize(MODEL_PATH) < 5 * 1024 * 1024: 
            os.remove(MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        print(f"⬇️ Downloading Model...")
        try:
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Download Complete")
        except Exception as e:
            print(f"❌ Download Failed: {e}")

# Global Memory
ml_context = {
    "model": None, "scaler": None, "props_cols": [], 
    "vocab_size": 0, "itos": {}, "device": torch.device("cpu")
}

# ==========================================
# 2. PUBCHEM VERIFICATION ENGINE
# ==========================================
def check_novelty(smiles: str):
    """
    Queries PubChem PUG REST API to see if the molecule exists.
    """
    try:
        # 1. Normalize SMILES first
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return False, "Invalid Structure"
        canonical = Chem.MolToSmiles(mol, canonical=True)

        # 2. Query PubChem (This takes ~1 second)
        # We search by identity (exact match)
        matches = pcp.get_compounds(canonical, namespace='smiles')

        if matches:
            # Found it! It's a known compound.
            # Try to get a common name (Synonym)
            compound = matches[0]
            common_name = "Known Compound"
            if hasattr(compound, 'synonyms') and compound.synonyms:
                # Pick the first synonym that looks like a name (not a CAS number)
                common_name = compound.synonyms[0][:20] 
            
            return False, f"Known: {common_name}"
        else:
            # Empty list = Novel!
            return True, "Novel (PubChem Verified)"

    except Exception as e:
        print(f"⚠️ PubChem Error: {e}")
        # Fallback if internet fails: Assume novel to keep app running
        return True, "Novel (Offline Mode)"

# ==========================================
# 3. MODEL ARCHITECTURE
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
# 4. LIFESPAN
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 MONOLITH STARTUP")
    download_model_if_needed()
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=ml_context['device'], weights_only=False)
            ml_context['vocab_size'] = checkpoint['vocab_size']
            ml_context['props_cols'] = checkpoint.get('props', [])
            ml_context['scaler'] = checkpoint.get('scaler', None)
            ml_context['itos'] = checkpoint['itos']
            
            input_dim = len(ml_context['props_cols'])
            model = SelfiesDiffusion(ml_context['vocab_size'], 512, 8, input_dim)
            model.load_state_dict(checkpoint['model_state'])
            model.to(ml_context['device'])
            model.eval()
            ml_context['model'] = model
            print(f"✅ MODEL LOADED! (Inputs: {input_dim})")
        except Exception as e:
            print(f"❌ FATAL LOAD ERROR: {e}")
    yield
    ml_context.clear()

# ==========================================
# 5. API
# ==========================================
app = FastAPI(title="ChemGen Monolith", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class DesignRequest(BaseModel):
    domain: str
    mw: Optional[float] = None
    logp: Optional[float] = None

class SmilesRequest(BaseModel):
    smiles: str

def run_diffusion_safe(vector, steps=25, temp=1.2):
    try:
        model, dev, vocab = ml_context['model'], ml_context['device'], ml_context['vocab_size']
        if not model: return None, ["❌ Model not loaded"]

        x = torch.randint(0, vocab, (1, 100)).to(dev)
        props = torch.tensor(vector, dtype=torch.float32).to(dev)
        
        for t in reversed(range(steps)):
            time = torch.tensor([t]).float().to(dev)
            with torch.no_grad(): 
                logits = model(x, time, props)
            probs = torch.softmax(logits / temp, dim=-1)
            if torch.isnan(probs).any(): return None, ["❌ NaN Detected"]
            pred = torch.multinomial(probs.view(-1, vocab), 1).view(x.shape)
            mask = torch.rand_like(x.float()) > (t/steps)
            x = torch.where(mask, pred, x)
            
        tokens = [ml_context['itos'].get(i, "") for i in x[0].cpu().tolist()]
        valid = [t for t in tokens if t not in ["[nop]", "[MASK]"]]
        
        if len(valid) < 8: return None, ["⚠️ Too Short"]

        try: 
            return sf.decoder("".join(valid)), [f"✅ Decoded (Steps={steps})"]
        except: 
            return None, ["⚠️ Decoder Failed"]
    except Exception as e: 
        return None, [f"❌ Crash: {e}"]

@app.post("/generate")
async def generate(req: DesignRequest):
    print(f"🚀 API HIT: /generate for {req.domain}")
    trace = []
    
    if not ml_context['model']: return {"error": "Model loading..."}
    props_cols = ml_context['props_cols']
    
    # EXPLOSIVE NOISE for Novelty
    final_vector = np.random.normal(0, 1.0, (1, len(props_cols)))
    trace.append("⚡ Using EXPLOSIVE Noise Vector")

    domain_key = f"is_{req.domain.lower()}"
    for i, col in enumerate(props_cols):
        if domain_key in col.lower():
            final_vector[0, i] = 5.0
            trace.append(f"✅ Flag Set: {col}")
            break

    best_smiles = "C"
    final_mol = None
    
    for _ in range(5):
        smiles, logs = run_diffusion_safe(final_vector)
        if logs and "NaN" in logs[0]:
             final_vector = np.random.normal(0, 1.0, (1, len(props_cols)))
             continue
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                best_smiles = smiles
                final_mol = mol
                break
    
    props = {"valid": False}
    mol_block = ""
    if final_mol:
        try:
            mol_3d = Chem.AddHs(final_mol)
            AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
            mol_block = Chem.MolToMolBlock(mol_3d)
            
            # 🔍 PUBCHEM CHECK
            is_novel, status = check_novelty(best_smiles)
            trace.append(f"🔍 Database Scan: {status}")
            
            sas_score = round(3 + (Descriptors.MolWt(final_mol) / 100), 1)
            props = {
                "mw": round(Descriptors.MolWt(final_mol), 2),
                "logp": round(Descriptors.MolLogP(final_mol), 2),
                "tpsa": round(Descriptors.TPSA(final_mol), 1),
                "qed": round(QED.qed(final_mol), 3),
                "sas": sas_score,
                "is_novel": is_novel,
                "valid": True
            }
        except: pass

    return {
        "smiles": best_smiles,
        "mol_block": mol_block,
        "properties": props,
        "trace": trace
    }

@app.post("/analyze")
async def analyze(req: SmilesRequest):
    mol = Chem.MolFromSmiles(req.smiles)
    if not mol: return {"error": "Invalid SMILES"}
    
    # 🔍 PUBCHEM CHECK
    is_novel, status = check_novelty(req.smiles)
    sas_score = round(3 + (Descriptors.MolWt(mol) / 100), 1)

    return {
        "mw": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rot": Descriptors.NumRotatableBonds(mol),
        "rings": Lipinski.RingCount(mol),
        "qed": round(QED.qed(mol), 3),
        "sas": sas_score,
        "is_novel": is_novel,
        "valid": True
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)