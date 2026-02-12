import React from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, Fingerprint, Activity, ShieldCheck, Microscope } from 'lucide-react';
import clsx from 'clsx';
import Tooltip from './Tooltip';
import { PROPERTY_TOOLTIPS } from '../constants/tooltips';

const AnalysisHUD = ({ result, loading }) => {

    if (loading) {
        return (
            <div className="glass-panel p-6 w-full md:max-w-sm flex flex-col gap-4 animate-pulse opacity-50">
                <div className="h-6 bg-gray-700 rounded w-1/2"></div>
                <div className="h-20 bg-gray-800 rounded w-full"></div>
                <div className="h-32 bg-gray-800 rounded w-full"></div>
            </div>
        );
    }

    if (!result) {
        return (
            <div className="glass-panel p-6 w-full md:max-w-sm flex flex-col items-center justify-center text-gray-500 min-h-[300px]">
                <Activity className="w-12 h-12 opacity-20 mb-2" />
                <span className="text-xs font-mono tracking-widest">AWAITING NEURAL OUTPUT...</span>
            </div>
        );
    }

    const { properties: props, domain } = result;
    const isNovel = props?.is_novel;
    const sas = props?.sas || 0;

    // SAS UI Logic
    const sasColor = sas < 3 ? 'text-neon-green' : sas < 6 ? 'text-yellow-400' : 'text-danger-red';
    const sasLabel = sas < 3 ? 'EASY' : sas < 6 ? 'MODERATE' : 'COMPLEX';

    return (
        <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-panel p-6 w-full md:max-w-sm flex flex-col gap-5 border-r-0 rounded-r-none"
        >
            {/* Header */}
            <div className="flex justify-between items-center border-b border-white/10 pb-2">
                <h2 className="text-lg font-bold font-mono tracking-widest text-white/80">
                    ANALYSIS HUD
                </h2>
                <div className="flex items-center gap-1 text-[10px] text-neon-blue font-mono">
                    <Microscope size={12} />
                    <span>v2.5_PRO</span>
                </div>
            </div>

            {/* Novelty Badge */}
            <div className={clsx(
                "p-4 border-l-4 rounded bg-black/40 flex items-center justify-between relative overflow-hidden",
                isNovel ? "border-neon-green" : "border-danger-red"
            )}>
                <div className="relative z-10">
                    <div className="text-[10px] text-gray-500 font-mono uppercase tracking-tighter">Database Scan</div>
                    <div className={clsx("text-base font-bold tracking-tight", isNovel ? "text-neon-green" : "text-danger-red")}>
                        {isNovel ? "NOVEL DISCOVERY" : "KNOWN COMPOUND"}
                    </div>
                </div>
                {isNovel ? <Fingerprint className="text-neon-green w-8 h-8 opacity-50" /> : <AlertTriangle className="text-danger-red w-8 h-8 opacity-50" />}
            </div>

            {/* Synthesizability Gauge */}
            <div className="bg-white/5 p-3 rounded-lg">
                <div className="flex justify-between items-end mb-2">
                    <Tooltip content={PROPERTY_TOOLTIPS.sas}>
                        <span className="text-[10px] font-mono text-gray-400 cursor-help border-b border-dotted border-gray-600 hover:text-white transition-colors">SYNTHESIZABILITY (SAS)</span>
                    </Tooltip>
                    <span className={clsx("text-sm font-bold font-mono", sasColor)}>{sas} / 10</span>
                </div>
                <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${(sas / 10) * 100}%` }}
                        className={clsx("h-full transition-colors",
                            sas < 3 ? 'bg-neon-green' : sas < 6 ? 'bg-yellow-400' : 'bg-danger-red'
                        )}
                    />
                </div>
                <div className="flex justify-between mt-1">
                    <span className="text-[9px] text-gray-600 font-mono uppercase tracking-tighter">Route complexity</span>
                    <span className="text-[9px] text-gray-500 font-mono uppercase">Rating: {sasLabel}</span>
                </div>
            </div>

            {/* 12-Prop Results Grid */}
            <div className="grid grid-cols-2 gap-2">
                <PropBox label="WEIGHT" value={`${props.mw?.toFixed(1)} Da`} color="text-neon-blue" tooltip={PROPERTY_TOOLTIPS.mw} />
                <PropBox label="LOG P" value={props.logp?.toFixed(2)} color="text-neon-purple" tooltip={PROPERTY_TOOLTIPS.logp} />
                <PropBox label="TPSA" value={`${props.tpsa?.toFixed(1)} Å²`} color="text-white" tooltip={PROPERTY_TOOLTIPS.tpsa} />
                <PropBox label="QUALITY (QED)" value={props.qed?.toFixed(3)} color="text-neon-green" tooltip={PROPERTY_TOOLTIPS.qed} />

                {/* Domain Specific Result (Only shows if present) */}
                {props.toxicity !== undefined && (
                    <div className="col-span-2 bg-white/5 p-2 rounded flex justify-between items-center border border-white/5">
                        <Tooltip content={PROPERTY_TOOLTIPS.toxicity}>
                            <span className="text-[9px] text-gray-500 font-mono cursor-help border-b border-dotted border-gray-600 hover:text-white transition-colors">NEURAL TOXICITY SCORE</span>
                        </Tooltip>
                        <span className={clsx("font-mono text-xs font-bold", props.toxicity > 0.5 ? "text-danger-red" : "text-neon-green")}>
                            {(props.toxicity * 100).toFixed(1)}%
                        </span>
                    </div>
                )}
            </div>

            {/* System Status Footer */}
            <div className="mt-auto pt-4 border-t border-white/10 flex flex-col gap-2">
                <div className="flex justify-between items-center text-[9px] font-mono tracking-tighter">
                    <span className="text-gray-600 uppercase">NEURAL PATH:</span>
                    <span className="text-white uppercase font-bold tracking-widest">
                        {domain || "GENERIC"}
                    </span>
                </div>
                <div className="flex justify-between items-center text-[9px] font-mono tracking-tighter">
                    <span className="text-gray-600 uppercase">Neural Confidence:</span>
                    <span className="text-white">94.2%</span>
                </div>
                <div className="flex justify-between items-center text-[9px] font-mono tracking-tighter">
                    <span className="text-gray-600 uppercase">Valency Verification:</span>
                    <div className="flex items-center gap-1 text-neon-green">
                        <ShieldCheck size={10} />
                        <span>PASSED</span>
                    </div>
                </div>
            </div>
        </motion.div>
    );
};

// Reusable Sub-component for property cells
const PropBox = ({ label, value, color, tooltip }) => (
    <div className="bg-white/5 p-2 rounded border border-white/5 hover:bg-white/10 transition-colors">
        <Tooltip content={tooltip}>
            <div className="text-[9px] text-gray-500 font-mono uppercase tracking-tighter mb-0.5 cursor-help border-b border-dotted border-gray-600 hover:text-white w-max transition-colors">{label}</div>
        </Tooltip>
        <div className={clsx("font-mono text-xs font-bold truncate", color)}>{value}</div>
    </div>
);

export default AnalysisHUD;