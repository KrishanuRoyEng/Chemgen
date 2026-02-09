import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
// Ensure Search is imported here
import { Beaker, Zap, Brain, Sliders, ChevronDown, ChevronUp, Search } from 'lucide-react';
import clsx from 'clsx';

const ControlDeck = ({
    domain, setDomain, mw, setMw, logp, setLogp,
    tpsa, setTpsa, hbd, setHbd, hba, setHba,
    rings, setRings, qed, setQed, toxicity, setToxicity,
    rot, setRot,
    adhesion, setAdhesion, affinity, setAffinity,
    onGenerate, loading,
    searchQuery, setSearchQuery, onReferenceSearch,
    primaryLead, setPrimaryLead
}) => {
    const [showAdvanced, setShowAdvanced] = React.useState(false);

    const domains = [
        { id: 'drug', label: 'Drug Discovery', icon: Beaker, color: 'text-neon-blue', border: 'border-neon-blue' },
        { id: 'material', label: 'Material Sci', icon: Zap, color: 'text-neon-purple', border: 'border-neon-purple' },
        { id: 'bio', label: 'Bio-Defense', icon: Brain, color: 'text-neon-green', border: 'border-neon-green' },
    ];

    const activeDomain = domains.find(d => d.id === domain);

    return (
        <div className="glass-panel p-6 w-full md:max-w-md flex flex-col gap-5 relative overflow-hidden">
            <h2 className="text-xl font-bold font-mono tracking-widest text-white/80 border-b border-white/10 pb-2">
                CONTROL DECK
            </h2>

            {/* REFERENCE SEARCH & NEURAL BIAS */}
            <div className="space-y-3 mb-6">
                <div className="flex gap-2">
                    <div className="relative flex-1">
                        <input
                            type="text"
                            placeholder="Aspirin, Caffeine, Lidocaine..."
                            className="w-full bg-cyber-black border border-white/10 rounded px-3 py-2 text-[10px] font-mono focus:border-neon-blue outline-none placeholder:text-gray-700"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && onReferenceSearch()}
                        />
                        <Search className="absolute right-3 top-2.5 text-gray-600" size={12} />
                    </div>
                    <button
                        onClick={onReferenceSearch}
                        className="bg-neon-blue/10 border border-neon-blue/30 px-4 rounded hover:bg-neon-blue/20 transition-all text-neon-blue font-bold text-[10px] uppercase font-mono"
                    >
                        SEED
                    </button>
                </div>

                {/* BIAS SELECTOR: Chemists click these to set the 'Anchor' molecule */}
                <div className="flex flex-wrap gap-2 min-h-[24px]">
                    {searchQuery.split(',').filter(n => n.trim() !== "").map((name, idx) => (
                        <motion.button
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            key={idx}
                            onClick={() => setPrimaryLead(idx)}
                            className={clsx(
                                "text-[8px] px-2 py-1 rounded border transition-all font-mono uppercase tracking-tighter flex items-center gap-1",
                                primaryLead === idx
                                    ? "bg-neon-blue/20 border-neon-blue text-neon-blue shadow-[0_0_10px_rgba(0,163,255,0.2)]"
                                    : "border-white/5 text-gray-500 hover:border-white/20"
                            )}
                        >
                            {primaryLead === idx && <Zap size={8} className="fill-current" />}
                            {name.trim() || "Untitled"}
                        </motion.button>
                    ))}
                </div>

                <p className="text-[9px] text-gray-600 font-mono uppercase tracking-tighter">
                    {searchQuery.includes(',')
                        ? "Click a compound to set as Neural Anchor (80% Bias)"
                        : "Enter multiple leads for property interpolation"}
                </p>
            </div>

            {/* DOMAIN SWITCHER */}
            <div className="flex gap-2 p-1 bg-cyber-black/50 rounded-lg">
                {domains.map((d) => (
                    <button
                        key={d.id}
                        onClick={() => setDomain(d.id)}
                        className={clsx(
                            "flex-1 py-3 px-2 rounded-md transition-all duration-300 flex flex-col items-center gap-1 text-[10px] font-bold uppercase",
                            domain === d.id ? `bg-white/10 ${d.color} shadow-lg` : "text-gray-500 hover:text-white"
                        )}
                    >
                        <d.icon size={18} />
                        {d.label}
                    </button>
                ))}
            </div>

            <div className="space-y-5 overflow-y-auto pr-2 max-h-[450px] custom-scrollbar">
                <div className="space-y-4">
                    <Slider label="MOLECULAR WEIGHT" value={mw} min={100} max={800} unit=" Da" color="accent-neon-blue" onChange={setMw} />
                    <Slider label="LOG P (SOLUBILITY)" value={logp} min={-3} max={6} step={0.1} color="accent-neon-blue" onChange={setLogp} />
                </div>

                <div className="pt-2">
                    <button onClick={() => setShowAdvanced(!showAdvanced)} className="flex items-center gap-2 text-[10px] font-mono text-gray-500 hover:text-white uppercase tracking-widest mb-4">
                        <Sliders size={12} />
                        {showAdvanced ? "Hide Neural Constraints" : "Show Neural Constraints"}
                        {showAdvanced ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                    </button>

                    <AnimatePresence>
                        {showAdvanced && (
                            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="space-y-4 border-t border-white/5 pt-4 overflow-hidden">
                                <Slider label="RINGS" value={rings} min={0} max={6} color="accent-white" onChange={setRings} />
                                <Slider label="QED (QUALITY)" value={qed} min={0} max={1} step={0.01} color="accent-neon-blue" onChange={setQed} />
                                <Slider label="TPSA (POLARITY)" value={tpsa} min={0} max={200} unit=" Å²" color="accent-white" onChange={setTpsa} />
                                <Slider label="TOXICITY LIMIT" value={toxicity} min={0} max={1} step={0.01} color="accent-danger-red" onChange={setToxicity} />
                                <Slider label="FLEXIBILITY (ROT)" value={rot} min={0} max={12} color="accent-white" onChange={setRot} />
                                <div className="grid grid-cols-2 gap-4">
                                    <Slider label="H-BOND DONOR" value={hbd} min={0} max={10} color="accent-gray-500" onChange={setHbd} />
                                    <Slider label="H-BOND ACCPT" value={hba} min={0} max={10} color="accent-gray-500" onChange={setHba} />
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                <motion.div key={domain} className={clsx("p-4 rounded-lg border border-dashed", activeDomain.border)}>
                    {domain === 'drug' && <div className="text-center text-[10px] text-gray-400 font-mono">TARGET: GENERIC SMALL MOLECULE<br />STATUS: <span className="text-neon-green">ACTIVE</span></div>}
                    {domain === 'material' && <Slider label="ADHESION STRENGTH" value={adhesion} min={0} max={10} step={0.5} unit=" / 10" color="accent-neon-purple" onChange={setAdhesion} />}
                    {domain === 'bio' && <Slider label="BINDING AFFINITY" value={affinity} min={4} max={12} step={0.5} unit=" -logKd" color="accent-neon-green" onChange={setAffinity} />}
                </motion.div>
            </div>

            <button onClick={onGenerate} disabled={loading} className={clsx("w-full py-4 mt-2 font-bold tracking-widest uppercase transition-all duration-300 relative overflow-hidden group border", loading ? "bg-gray-800 border-gray-700 cursor-not-allowed text-gray-500" : `bg-cyber-black ${activeDomain.border} ${activeDomain.color} hover:bg-white/5`)}>
                <span className="relative z-10 flex items-center justify-center gap-2">
                    {loading ? "SYNTHESIZING..." : "INITIATE SYNTHESIS"}
                </span>
            </button>
        </div>
    );
};

const Slider = ({ label, value, min, max, step = 1, unit = "", color, onChange }) => (
    <div className="space-y-2">
        <div className="flex justify-between text-[10px] font-mono text-white/50">
            <span>{label}</span><span className="text-white">{value}{unit}</span>
        </div>
        <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} className={clsx("w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer", color)} />
    </div>
);

export default ControlDeck;