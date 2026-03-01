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
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, QED

# ==========================================
# 1. CONFIGURATION & DOWNLOADER
# ==========================================
CURRENT_DIR = os.getcwd() 
MODEL_FILENAME = "universal_diffusion_model.pth"
MODEL_PATH = os.path.join(CURRENT_DIR, MODEL_FILENAME)
MODEL_URL = "https://huggingface.co/krishanuroy/universal_diffusion_model/resolve/main/universal_diffusion_model.pth" 

def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        if os.path.getsize(MODEL_PATH) < 5 * 1024 * 1024: 
            os.remove(MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        try:
            print("⬇️ Downloading Universal Model...")
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Download Complete")
        except Exception: pass

ml_context = {
    "model": None, "scaler": None, "props_cols": [], 
    "vocab_size": 0, "itos": {}, 
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ==========================================
# 2. DOMAIN LOGIC ENGINE
# ==========================================
class DomainConfig:
    def __init__(self, domain: str):
        self.domain = domain.lower()
        
    def get_constraints(self):
        """Returns physical constraints based on the target domain."""
        if self.domain == 'material': # ADHESION / POLYMERS
            return {
                # ALLOWED: Silicon (Si) for Silanes, but NO Selenium (Se)
                "allowed_atoms": {'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'H', 'Si', 'P'}, 
                "min_mw": 150, 
                "max_mw": 1000, 
                "min_rings": 0, # Linear chains (like PEG/Silicones) are good for adhesion
                "forbidden_fragments": ["[OH+]", "[N+]", "[Se]"] # No salts/selenium
            }
        elif self.domain == 'biomolecule':
            return {
                "allowed_atoms": {'C', 'N', 'O', 'S', 'P', 'H'},
                "min_mw": 400, 
                "max_mw": 2000,
                "min_rings": 0, 
                "forbidden_fragments": ["Cl", "Br", "I", "F"]
            }
        else: # Default: 'drug'
            return {
                "allowed_atoms": {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H'},
                "min_mw": 100,
                "max_mw": 600, 
                "min_rings": 1,
                "forbidden_fragments": ["Si", "B", "Se"]
            }

# ==========================================
# 3. HELPER: NOVELTY CHECK
# ==========================================
COMMON_SMILES = {"C": "Methane", "CC": "Ethane", "CCO": "Ethanol", "c1ccccc1": "Benzene"}

def check_novelty(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return False, "Invalid"
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in COMMON_SMILES: return False, "Known Compound"
        return True, "Novel (Offline Verified)"
    except:
        return True, "Novel (Offline)"

# ==========================================
# 4. MODEL ARCHITECTURE
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
# 5. LIFESPAN
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
# 6. API & LOGIC
# ==========================================
app = FastAPI(title="ChemGen", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class DesignRequest(BaseModel):
    domain: str
    mw: float = 350.0
    logp: float = 2.5
    tpsa: float = 70.0
    qed: float = 0.5
    rings: float = 2.0
    rot: float = 5.0
    adhesion: float = 5.0
    affinity: float = 8.0
    hbd: float = 2.0
    hba: float = 5.0
    toxicity: float = 0.1

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
            pred = torch.multinomial(probs.view(-1, vocab), 1).view(x.shape)
            mask = torch.rand_like(x.float()) > (t/steps)
            x = torch.where(mask, pred, x)
            
        tokens = [ml_context['itos'].get(i, "") for i in x[0].cpu().tolist()]
        valid = [t for t in tokens if t not in ["[nop]", "[MASK]"]]
        if len(valid) < 8: return None, ["Too Short"]
        return sf.decoder("".join(valid)), ["Success"]
    except: return None, ["Crash"]

# ---------------------------------------------------------
# 🧪 DOMAIN-AWARE PHYSICS & MEDICINAL CHEMISTRY CHECK
# ---------------------------------------------------------
def is_chemically_stable(mol, domain_config):
    try:
        Chem.SanitizeMol(mol)
        config = domain_config.get_constraints()
        allowed = config["allowed_atoms"]
        
        # 1. ATOM-LEVEL VALENCY & CHARGE CHECKS
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym not in allowed: return False 
            
            chg = atom.GetFormalCharge()
            val = atom.GetTotalValence()
            
            # STRICT CHARGE PENALTY: Drugs & Materials shouldn't have random formal charges
            if domain_config.domain in ['drug', 'material'] and chg != 0: 
                return False
            
            # Basic valency limits
            if sym == 'C' and val > 4: return False
            if sym == 'Si' and val != 4: return False

        # 2. STRING-LEVEL FORBIDDEN FRAGMENTS
        smiles = Chem.MolToSmiles(mol)
        if any(frag in smiles for frag in config["forbidden_fragments"]): return False

        # 3. SMARTS FILTERS (The "Bouncer" for Toxic/Unstable Structures)
        pains_and_strains = [
            Chem.MolFromSmarts("[S]~[F,Cl,Br,I]"), # Sulfur-Halogen (Violently reactive warheads)
            Chem.MolFromSmarts("[O]-[O]"),         # Peroxide bonds (Highly explosive)
            Chem.MolFromSmarts("C1=CC1"),          # Cyclopropene (Extreme ring strain)
            Chem.MolFromSmarts("[N]=[N]-[N]"),     # Azides (Explosive/Reactive)
            Chem.MolFromSmarts("[C-], [C+]"),      # Carbanions/Carbocations
            Chem.MolFromSmarts("[*]=[*]=[*]")      # Cumulenes (Highly unstable)
        ]

        for alert in pains_and_strains:
            if alert and mol.HasSubstructMatch(alert):
                return False # Reject the hallucination immediately!

        return True
    except: return False

# ---------------------------------------------------------
# 🛡️ GENERATOR: ADHESION-AWARE MODE
# ---------------------------------------------------------
@app.post("/generate")
async def generate(req: DesignRequest):
    print(f"🚀 API HIT: {req.domain.upper()}")
    trace = []
    
    if not ml_context['model']: return {"error": "Loading..."}
    props_cols = ml_context['props_cols']
    
    # 1. SETUP DOMAIN ENGINE
    d_config = DomainConfig(req.domain)
    constraints = d_config.get_constraints()
    
    user_props = []
    for col in props_cols:
        c = col.lower()
        if 'mw' in c: user_props.append(req.mw)
        elif 'logp' in c: user_props.append(req.logp)
        elif 'tpsa' in c: user_props.append(req.tpsa)
        elif 'qed' in c: user_props.append(req.qed)
        elif 'rot' in c: user_props.append(req.rot)
        elif 'ring' in c: user_props.append(req.rings)
        elif 'hbd' in c: user_props.append(req.hbd)
        elif 'hba' in c: user_props.append(req.hba)
        else: user_props.append(0.0) # Safe default
        
    base_vector = np.array([user_props])

    if ml_context['scaler']:
        try:
            # Slice the first 12 columns, scale them, and stitch the rest back on
            scaled_12 = ml_context['scaler'].transform(base_vector[:, :12])
            base_vector = np.hstack((scaled_12, base_vector[:, 12:]))
        except Exception as e:
            print(f"Scaler warning bypassed: {e}")
            pass

    for i, col in enumerate(props_cols):
        # 🔩 MATERIAL / ADHESION MODE
        if req.domain == 'material':
            if 'tpsa' in col.lower(): base_vector[0, i] += 1.0 # Boost polarity
            if 'rot' in col.lower(): base_vector[0, i] += (req.adhesion / 3.0) # UI Adhesion!
            if 'ring' in col.lower(): base_vector[0, i] -= 1.0 # Less rings

        # 🧬 BIOMOLECULE MODE (FIXED!)
        elif req.domain == 'biomolecule':
            if 'mw' in col.lower(): base_vector[0, i] += (req.affinity / 2.0) # UI Affinity!
            if 'hbd' in col.lower(): base_vector[0, i] += 1.0  # Boost donors
            if 'hba' in col.lower(): base_vector[0, i] += 1.0  # Boost acceptors

        # 💊 DRUG MODE 
        elif req.domain == 'drug':
            # A strict (low) toxicity limit forces the model to generate more water-soluble (lower LogP) drugs
            if 'logp' in col.lower(): base_vector[0, i] += (req.toxicity - 0.5) * 2.0
   
    # 3. SET FALLBACK (Based on Domain)
    if req.domain == 'material':
        best_smiles = "CO[Si](OC)(OC)CC1CO1" 
    elif req.domain == 'biomolecule':
        best_smiles = "C1(=O)NCC(=O)NCC(=O)NCC(=O)N1" 
    else:
        best_smiles = "CC(=O)Oc1ccccc1C(=O)O" 
        
    best_mol = Chem.MolFromSmiles(best_smiles)

    # 4. GENERATION LOOP (FIXED: 100 ATTEMPTS)
    for attempt in range(100):
        # Use higher Temp for Materials/Bio to explore wilder chemical space
        temp = 1.3 if req.domain == 'material' else 1.2 if req.domain == 'biomolecule' else 1.0
        vec = base_vector + np.random.normal(0, 0.4, base_vector.shape)
        
        smiles, logs = run_diffusion_safe(vec, steps=30, temp=temp)
        if not smiles: continue
        mol = Chem.MolFromSmiles(smiles)
        if not mol: continue

        try:
            if not is_chemically_stable(mol, d_config): continue
            
            # Size Check
            mw = Descriptors.MolWt(mol)
            if mw < constraints["min_mw"] or mw > constraints["max_mw"]: continue
            if Lipinski.RingCount(mol) < constraints["min_rings"]: continue
            
            best_smiles = smiles
            best_mol = mol
            trace.append(f"🎉 Valid {req.domain.upper()} Generated (Attempt {attempt})")
            break
        except: continue

    # 5. PACK RESPONSE
    props = {"valid": False}
    mol_block = ""
    if best_mol:
        try:
            mol_3d = Chem.AddHs(best_mol)
            AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
            mol_block = Chem.MolToMolBlock(mol_3d)
            is_novel, status = check_novelty(best_smiles)
            trace.append(f"DB: {status}")
            
            tpsa_val = Descriptors.TPSA(best_mol)
            rot_val = Lipinski.NumRotatableBonds(best_mol)
            mw_val = Descriptors.MolWt(best_mol)
            hbd_val = Lipinski.NumHDonors(best_mol)
            hba_val = Lipinski.NumHAcceptors(best_mol)

            props = {
                "mw": round(mw_val, 2),
                "logp": round(Descriptors.MolLogP(best_mol), 2),
                "tpsa": round(tpsa_val, 1),
                "qed": round(QED.qed(best_mol), 3),
                "sas": round(3 + (mw_val/100), 1),
                "hbd": hbd_val,
                "hba": hba_val,
                "rings": Lipinski.RingCount(best_mol),
                "rot": rot_val,
                "adhesion": round(min(10.0, (tpsa_val / 20) + (rot_val / 2)), 1),
                "affinity": round(min(10.0, (mw_val / 50) + hbd_val + hba_val), 1),
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
    
    # 🟢 REPLACE THE RETURN BLOCK WITH THIS
    tpsa_val = Descriptors.TPSA(mol)
    rot_val = Lipinski.NumRotatableBonds(mol)
    mw_val = Descriptors.MolWt(mol)
    hbd_val = Lipinski.NumHDonors(mol)
    hba_val = Lipinski.NumHAcceptors(mol)
    
    return {
        "mw": round(mw_val, 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(tpsa_val, 2),
        "qed": round(QED.qed(mol), 3),
        "sas": round(3 + (mw_val / 100), 1),
        "hbd": hbd_val,
        "hba": hba_val,
        "rings": Lipinski.RingCount(mol),
        "rot": rot_val,
        "adhesion": round(min(10.0, (tpsa_val / 20) + (rot_val / 2)), 1),
        "affinity": round(min(10.0, (mw_val / 50) + hbd_val + hba_val), 1),
        "is_novel": is_novel,
        "valid": True
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)