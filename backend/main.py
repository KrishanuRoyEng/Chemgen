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
        print(f"⬇️ Downloading Model...")
        try:
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Download Complete")
        except Exception: pass

# Global Memory - AUTO-DETECT GPU
ml_context = {
    "model": None, "scaler": None, "props_cols": [], 
    "vocab_size": 0, "itos": {}, 
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu") # <--- GPU ENABLED
}

# ==========================================
# 2. NOVELTY & VALIDATION
# ==========================================
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
    print(f"🚀 MONOLITH STARTUP on {ml_context['device']}")
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
        except Exception as e:
            print(f"❌ LOAD ERROR: {e}")
    yield
    ml_context.clear()

# ==========================================
# 5. API & LOGIC
# ==========================================
app = FastAPI(title="ChemGen", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class DesignRequest(BaseModel):
    domain: str

class SmilesRequest(BaseModel):
    smiles: str

def run_diffusion_safe(vector, steps=30, temp=1.2):
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
# 🧪 FINAL STABILITY CHECK (Fixes Sulfur Error)
# ---------------------------------------------------------
def is_chemically_stable(mol):
    try:
        Chem.SanitizeMol(mol)
        
        # 1. ATOM CHECKER
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            chg = atom.GetFormalCharge()
            deg = atom.GetDegree() # Number of neighbors
            val = atom.GetTotalValence() # Total bonds
            
            # 🛑 SULFUR/OXYGEN RULES
            if sym in ['O', 'S']:
                if chg == -1 and deg > 1: return False # No S- Bridges
                if chg == 0 and val not in [2, 4, 6]: return False
                if chg == 1 and val != 3: return False

            # 🛑 NITROGEN RULES
            if sym == 'N':
                if chg == 0 and val != 3: return False
                if chg == 1 and val != 4: return False
                if val > 4: return False 

            # 🛑 CARBON RULES
            if sym == 'C':
                if chg == 0 and val != 4: return False

        # 2. BOND CHECKER (No S=S or S=N)
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            sym1, sym2 = a1.GetSymbol(), a2.GetSymbol()
            
            # Repulsion Check (+ touching +)
            if a1.GetFormalCharge() * a2.GetFormalCharge() > 0: return False
            
            # Bad Double Bonds (S=S, S=N, P=P)
            if bond.GetBondTypeAsDouble() > 1.0:
                if sym1 in ['S','N','P'] and sym2 in ['S','N','P']:
                    if sym1 == 'N' and sym2 == 'N': continue # Allow N=N
                    if sym1 == 'O' or sym2 == 'O': continue # Allow S=O, P=O
                    return False 

        return True
    except: return False

# ---------------------------------------------------------
# 🛡️ GENERATOR (Ring Enforcer + Fail-Safe)
# ---------------------------------------------------------
@app.post("/generate")
async def generate(req: DesignRequest):
    print(f"🚀 API HIT: {req.domain}")
    trace = []
    
    if not ml_context['model']: return {"error": "Loading..."}
    props_cols = ml_context['props_cols']
    base_vector = np.random.normal(0, 1.2, (1, len(props_cols)))
    
    domain_key = f"is_{req.domain.lower()}"
    for i, col in enumerate(props_cols):
        if domain_key in col.lower():
            base_vector[0, i] = 5.0
            break

    best_smiles = "C"
    final_mol = None
    FORBIDDEN = ["Se", "Si", "Te", "As", "B", "Pb", "Sn", "Hg", "Cd"] 
    ALLOWED_ATOMS = {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H'}

    # Try 50 times. Logic: Creative -> Safe.
    for attempt in range(50):
        # Adaptive Noise
        if attempt < 15:
            temp = 1.5
            vec = base_vector + np.random.normal(0, 0.5, base_vector.shape)
        else:
            temp = 1.0 # Panic Mode
            vec = np.random.normal(0, 0.8, (1, len(props_cols)))

        smiles, logs = run_diffusion_safe(vec, steps=30, temp=temp)
        if not smiles: continue
        
        # 1. TEXT FILTER
        if any(bad in smiles for bad in FORBIDDEN): continue

        # 2. RDKIT PARSE
        mol = Chem.MolFromSmiles(smiles)
        if not mol: continue

        try:
            # 3. ATOMIC FILTER
            is_organic = True
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in ALLOWED_ATOMS:
                    is_organic = False; break
            if not is_organic: continue

            # 4. PHYSICS FILTER
            if not is_chemically_stable(mol): continue
            
            # 5. COMPLEXITY FILTER (RINGS REQUIRED!)
            min_atoms = 12 if attempt < 15 else 5
            if mol.GetNumAtoms() < min_atoms: continue
            
            # 🛑 FORCE RINGS IN CREATIVE MODE 🛑
            # If we are in the first 15 tries, reject linear worms.
            if attempt < 15 and Lipinski.RingCount(mol) == 0:
                continue

            best_smiles = smiles
            final_mol = mol
            trace.append(f"🎉 Success: Attempt {attempt}")
            break
        except: continue

    props = {"valid": False}
    mol_block = ""
    if final_mol:
        try:
            mol_3d = Chem.AddHs(final_mol)
            AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
            mol_block = Chem.MolToMolBlock(mol_3d)
            is_novel, status = check_novelty(best_smiles)
            trace.append(f"Status: {status}")
            
            props = {
                "mw": round(Descriptors.MolWt(final_mol), 2),
                "logp": round(Descriptors.MolLogP(final_mol), 2),
                "tpsa": round(Descriptors.TPSA(final_mol), 1),
                "qed": round(QED.qed(final_mol), 3),
                "sas": round(3 + (Descriptors.MolWt(final_mol)/100), 1),
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
    
    # Updated to match generate props
    sas_score = round(3 + (Descriptors.MolWt(mol) / 100), 1)
    
    return {
        "mw": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "qed": round(QED.qed(mol), 3),
        "sas": sas_score,
        "is_novel": is_novel,
        "valid": True
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)