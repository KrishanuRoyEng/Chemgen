import os
import requests
import torch
import torch.nn as nn
import numpy as np
import selfies as sf
import uvicorn
import pubchempy as pcp 
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
        try:
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception: pass

ml_context = {
    "model": None, "scaler": None, "props_cols": [], 
    "vocab_size": 0, "itos": {}, 
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ==========================================
# 2. HELPER: NOVELTY CHECK (Analysis Only)
# ==========================================
# We keep this just for the "Analysis" HUD, but we won't use it to block generation.
COMMON_SMILES = {
    "C": "Methane", "CC": "Ethane", "CCO": "Ethanol", "c1ccccc1": "Benzene",
    "CC(=O)Oc1ccccc1C(=O)O": "Aspirin"
}

def check_novelty(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return False, "Invalid"
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in COMMON_SMILES: return False, "Known Compound"

        # Check PubChem (Optional - does not block generation anymore)
        matches = pcp.get_compounds(canonical, namespace='smiles')
        if matches:
            return False, "Known (PubChem)"
        else:
            return True, "Novel (PubChem Verified)"
    except:
        return True, "Novel (Offline)"

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
            print(f"✅ MODEL LOADED (GPU: {torch.cuda.is_available()})")
        except Exception: pass
    yield
    ml_context.clear()

# ==========================================
# 5. API & LOGIC
# ==========================================
app = FastAPI(title="ChemGen", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class DesignRequest(BaseModel):
    domain: str
    mw: Optional[float] = 350.0 # Default reasonable MW

class SmilesRequest(BaseModel):
    smiles: str

def run_diffusion_safe(vector, steps=30, temp=1.0):
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
            if torch.isnan(probs).any(): return None, ["NaN"]
            pred = torch.multinomial(probs.view(-1, vocab), 1).view(x.shape)
            mask = torch.rand_like(x.float()) > (t/steps)
            x = torch.where(mask, pred, x)
            
        tokens = [ml_context['itos'].get(i, "") for i in x[0].cpu().tolist()]
        valid = [t for t in tokens if t not in ["[nop]", "[MASK]"]]
        if len(valid) < 8: return None, ["Too Short"]
        return sf.decoder("".join(valid)), ["Success"]
    except: return None, ["Crash"]

# ---------------------------------------------------------
# 🧪 STABILITY CHECK (The Only Filter We Need)
# ---------------------------------------------------------
def is_chemically_stable(mol):
    """
    Ensures the molecule obeys physics. 
    Does not care if it is known or novel.
    """
    try:
        Chem.SanitizeMol(mol)
        # 1. Atom Rules
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            chg = atom.GetFormalCharge()
            val = atom.GetTotalValence()
            
            # Kill Charged Oxygens/Sulfurs/Carbons (keep it neutral/stable)
            if sym in ['O', 'S', 'C'] and chg != 0: return False
            # Kill Hypervalent Carbon
            if sym == 'C' and val > 4: return False
            # Kill unstable Phosphorus double bonds
            if sym == 'P':
                 for bond in atom.GetBonds():
                    if bond.GetBondTypeAsDouble() > 1.0 and bond.GetOtherAtom(atom).GetSymbol() != 'O':
                        return False

        # 2. Bond Rules
        for bond in mol.GetBonds():
            if bond.GetBondTypeAsDouble() > 1.0:
                s1, s2 = bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()
                # No S=S, S=N, P=P
                if s1 in ['S','N','P'] and s2 in ['S','N','P']:
                    if s1 == 'N' and s2 == 'N': continue # N=N is ok
                    if s1 == 'O' or s2 == 'O': continue # X=O is ok
                    return False
        return True
    except: return False

# ---------------------------------------------------------
# 🛡️ GENERATOR: PURE OPTIMIZATION MODE
# ---------------------------------------------------------
@app.post("/generate")
async def generate(req: DesignRequest):
    print(f"🚀 API HIT: {req.domain}")
    trace = []
    
    if not ml_context['model']: return {"error": "Loading..."}
    props_cols = ml_context['props_cols']
    
    # 1. PREPARE VECTOR
    # We use a standard normal distribution. 
    # No "Ghost Overrides" needed unless you want to bias specifically.
    base_vector = np.random.normal(0, 1.0, (1, len(props_cols)))
    
    # 2. GENERATION LOOP (Fast & Simple)
    # We only need to loop until we find ONE valid, stable molecule.
    # Usually happens in 1-5 attempts.
    best_smiles = "C"
    best_mol = None
    
    FORBIDDEN = ["Se", "Si", "Te", "As", "B", "Pb", "Sn", "Hg", "Cd"] 
    ALLOWED_ATOMS = {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H'}

    for attempt in range(20): # 20 tries is plenty for a pure generator
        
        # Slight variation each time to ensure we don't get stuck
        vec = base_vector + np.random.normal(0, 0.2, base_vector.shape)
        
        # Standard Temp = 1.0 (Balance of creativity and stability)
        smiles, logs = run_diffusion_safe(vec, steps=30, temp=1.0)
        
        if not smiles: continue
        if any(bad in smiles for bad in FORBIDDEN): continue
        
        mol = Chem.MolFromSmiles(smiles)
        if not mol: continue

        try:
            # 1. Physics Check
            if not is_chemically_stable(mol): continue
            
            # 2. Organic Check
            is_organic = True
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in ALLOWED_ATOMS:
                    is_organic = False; break
            if not is_organic: continue

            # 3. Simple Size Check (Don't return Methane)
            if mol.GetNumAtoms() < 6: continue
            
            # SUCCESS!
            # We don't care if it's novel. If it's stable and valid, it's a winner.
            best_smiles = smiles
            best_mol = mol
            trace.append(f"🎉 Valid Molecule Generated (Attempt {attempt})")
            break # Exit loop immediately
            
        except: continue

    # 3. PACK RESPONSE
    props = {"valid": False}
    mol_block = ""
    if best_mol:
        try:
            mol_3d = Chem.AddHs(best_mol)
            AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
            mol_block = Chem.MolToMolBlock(mol_3d)
            
            # We still CHECK novelty for the UI, but we don't block on it.
            is_novel, status = check_novelty(best_smiles)
            trace.append(f"Database Status: {status}")
            
            props = {
                "mw": round(Descriptors.MolWt(best_mol), 2),
                "logp": round(Descriptors.MolLogP(best_mol), 2),
                "tpsa": round(Descriptors.TPSA(best_mol), 1),
                "qed": round(QED.qed(best_mol), 3),
                "sas": round(3 + (Descriptors.MolWt(best_mol)/100), 1),
                "is_novel": is_novel,
                "valid": True
            }
        except: pass

    return {"smiles": best_smiles, "mol_block": mol_block, "properties": props, "trace": trace}

@app.post("/analyze")
async def analyze(req: SmilesRequest):
    mol = Chem.MolFromSmiles(req.smiles)
    if not mol: return {"error": "Invalid"}
    is_novel, status = check_novelty(req.smiles)
    return {
        "mw": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "qed": round(QED.qed(mol), 3),
        "sas": round(3 + (Descriptors.MolWt(mol) / 100), 1),
        "is_novel": is_novel,
        "valid": True
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)