import React, { useState } from "react";
import ControlDeck from "./components/ControlDeck";
import HolographicBox from "./components/HolographicBox";
import AnalysisHUD from "./components/AnalysisHUD";
import ComparisonTable from "./components/ComparisonTable";
import { Microscope } from "lucide-react";

function App() {
  // 1. DOMAIN & PRIMARY PROPS
  const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";
  const [domain, setDomain] = useState("drug");
  const [mw, setMw] = useState(350);
  const [logp, setLogp] = useState(2.5);

  // 2. NEURAL CONSTRAINTS
  const [tpsa, setTpsa] = useState(80);
  const [hbd, setHbd] = useState(2);
  const [hba, setHba] = useState(5);
  const [rings, setRings] = useState(2);
  const [qed, setQed] = useState(0.8);
  const [toxicity, setToxicity] = useState(0.1);
  const [rot, setRot] = useState(4);

  // 3. DOMAIN SPECIFIC
  const [adhesion, setAdhesion] = useState(5.0);
  const [affinity, setAffinity] = useState(7.0);

  // 4. MULTI-SEEDING & SEARCH
  const [processStatus, setProcessStatus] = useState("IDLE"); // 'IDLE' | 'SEEDING' | 'SEEDED' | 'SYNTHESIZING' | 'COMPLETE'
  const [result, setResult] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [primaryLead, setPrimaryLead] = useState(0);

  // --- MATH: WEIGHTED INTERPOLATION ---
  const handleReferenceSearch = async () => {
    if (!searchQuery) return;
    setProcessStatus("SEEDING");

    const names = searchQuery
      .split(",")
      .map((n) => n.trim())
      .filter((n) => n !== "");

    try {
      const allChemicalData = [];

      for (const name of names) {
        // Step A: Name to SMILES
        const pcRes = await fetch(
          `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${name}/property/SMILES/JSON`,
        );
        const pcData = await pcRes.json();
        if (!pcData.PropertyTable) continue;

        const smiles = pcData.PropertyTable.Properties[0].SMILES;

        // Step B: SMILES to 12 Props (Your Backend)
        const propRes = await fetch(`${API_URL}/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles }),
        });
        const props = await propRes.json();
        if (props.error) {
          console.warn(`Skipping ${name}: Backend could not parse SMILES`);
          continue;
        }

        allChemicalData.push(props);
      }

      if (allChemicalData.length === 0)
        throw new Error("No valid compounds found");

      // Step C: Weighted Logic
      // Primary Lead gets weight 1.0, others get 0.2
      const calculateWeighted = (key, useLog = false) => {
        let totalVal = 0;
        let totalWeight = 0;

        allChemicalData.forEach((data, idx) => {
          const weight = idx === primaryLead ? 1.0 : 0.2;
          const val = useLog ? Math.log10(data[key]) : data[key];
          totalVal += val * weight;
          totalWeight += weight;
        });

        const final = totalVal / totalWeight;
        return useLog ? Math.pow(10, final) : final;
      };

      // Update Sliders based on the "Center of Mass" of the chemicals
      setMw(Math.round(calculateWeighted("mw", true)));
      setLogp(parseFloat(calculateWeighted("logp").toFixed(2)));
      setTpsa(Math.round(calculateWeighted("tpsa")));
      setHbd(Math.round(calculateWeighted("hbd")));
      setHba(Math.round(calculateWeighted("hba")));
      setRings(Math.round(calculateWeighted("rings")));
      setRot(Math.round(calculateWeighted("rot")));
      setQed(parseFloat(calculateWeighted("qed").toFixed(3)));

      // Simulate a small delay for the "Seeding" visual effect to finish
      setTimeout(() => {
        setProcessStatus("SEEDED");
      }, 800);
    } catch (e) {
      console.error(e);
      alert("Lead Seeding Failed. Ensure names are correct.");
      setProcessStatus("IDLE");
    }
  };

  const handleGenerate = async () => {
    setProcessStatus("SYNTHESIZING");
    setResult(null);
    try {
      const payload = {
        domain,
        mw: parseFloat(mw),
        logp: parseFloat(logp),
        tpsa: parseFloat(tpsa),
        hbd: parseInt(hbd),
        hba: parseInt(hba),
        rings: parseInt(rings),
        qed: parseFloat(qed),
        toxicity: parseFloat(toxicity),
        rot: parseInt(rot),
        adhesion: parseFloat(adhesion),
        affinity: parseFloat(affinity),
      };

      const response = await fetch(`${API_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();

      if (!response.ok || data.detail || data.error) {
        throw new Error(data.error || "Backend validation failed");
      }

      console.log("🚀 BACKEND RESPONSE:", data); // DEBUG TRACE
      if (data.trace) console.log("📝 SERVER TRACE:", data.trace);

      setTimeout(() => {
        setResult(data);
        setProcessStatus("COMPLETE");
      }, 1500);
    } catch (e) {
      console.error(e);
      alert(`Synthesis Error: ${e.message}`);
      setProcessStatus("IDLE");
    }
  };

  return (
    <div className="min-h-screen bg-cyber-black text-white p-4 md:p-8 font-sans selection:bg-neon-blue selection:text-white">
      {/* Header Area */}
      <header className="flex justify-between items-center mb-8 border-b border-white/10 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-neon-blue/10 rounded-full border border-neon-blue/50 shadow-[0_0_15px_rgba(0,163,255,0.3)]">
            <Microscope className="text-neon-blue w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tighter">
              CHEMGEN{" "}
              <span className="text-neon-blue font-black italic">
                INTERPOLATOR
              </span>
            </h1>
            <p className="text-[10px] text-gray-500 font-mono tracking-widest uppercase">
              Lead-Optimization Engine v2.5
            </p>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* 1. LEFT: CONTROL DECK (Weighted Inputs) */}
        <div className="lg:col-span-4">
          <ControlDeck
            domain={domain}
            setDomain={setDomain}
            mw={mw}
            setMw={setMw}
            logp={logp}
            setLogp={setLogp}
            tpsa={tpsa}
            setTpsa={setTpsa}
            hbd={hbd}
            setHbd={setHbd}
            hba={hba}
            setHba={setHba}
            rings={rings}
            setRings={setRings}
            qed={qed}
            setQed={setQed}
            toxicity={toxicity}
            setToxicity={setToxicity}
            rot={rot}
            setRot={setRot}
            adhesion={adhesion}
            setAdhesion={setAdhesion}
            affinity={affinity}
            setAffinity={setAffinity}
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            primaryLead={primaryLead}
            setPrimaryLead={setPrimaryLead} // For Bias Selection
            onReferenceSearch={handleReferenceSearch}
            onGenerate={handleGenerate}
            processStatus={processStatus}
          />
        </div>

        {/* 2. CENTER: VISUALIZER & COMPARISON */}
        <div className="lg:col-span-5 flex flex-col justify-start gap-4">
          <HolographicBox
            processStatus={processStatus}
            smiles={result?.smiles}
            result={result}
          />

          {/* Neural Convergence Report (Comparison Table) */}
          {result && (
            <ComparisonTable
              domain={domain}
              targets={{
                mw,
                logp,
                tpsa,
                qed,
                hbd,
                hba,
                rings,
                rot,
                adhesion,
                affinity,
              }}
              result={result}
            />
          )}

          <div className="mt-2 text-center">
            <span className="text-gray-600 font-mono text-[10px] tracking-widest uppercase">
              {processStatus === "SYNTHESIZING"
                ? "Sampling Latent Space..."
                : result
                  ? "Property Match Confirmed"
                  : "Awaiting Seeding Input"}
            </span>
          </div>
        </div>

        {/* 3. RIGHT: FINAL ANALYSIS HUD */}
        <div className="lg:col-span-3">
          <AnalysisHUD
            result={result}
            loading={processStatus === "SYNTHESIZING"}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
