import React from 'react';
import clsx from 'clsx';
import Tooltip from './Tooltip';
import { PROPERTY_TOOLTIPS } from '../constants/tooltips';

const ComparisonTable = ({ targets, result }) => {
    if (!result || !result.properties) return null;
    const actual = result.properties;

    const metrics = [
        { label: 'Weight', target: targets.mw, actual: actual.mw, unit: 'Da', tooltip: PROPERTY_TOOLTIPS.mw },
        { label: 'LogP', target: targets.logp, actual: actual.logp, unit: '', tooltip: PROPERTY_TOOLTIPS.logp },
        { label: 'TPSA', target: targets.tpsa, actual: actual.tpsa, unit: 'Å²', tooltip: PROPERTY_TOOLTIPS.tpsa },
        { label: 'QED', target: targets.qed, actual: actual.qed, unit: '', tooltip: PROPERTY_TOOLTIPS.qed },
    ];

    return (
        <div className="mt-4 border-t border-white/10 pt-4">
            <h3 className="text-[10px] font-mono text-gray-500 mb-2 uppercase tracking-widest">Convergence Report</h3>
            <div className="space-y-1">
                {metrics.map((m) => {
                    const diff = Math.abs(m.target - m.actual);
                    const percent = ((diff / m.target) * 100).toFixed(1);
                    return (
                        <div key={m.label} className="flex justify-between items-center text-[10px] font-mono">
                            <Tooltip content={m.tooltip}>
                                <span className="text-gray-400 cursor-help border-b border-dotted border-gray-600 hover:text-gray-200 transition-colors">{m.label}</span>
                            </Tooltip>
                            <div className="flex gap-3">
                                <span className="text-gray-600">Tar: {m.target}</span>
                                <span className="text-white">Gen: {m.actual}</span>
                                <span className={clsx(percent < 15 ? "text-neon-green" : "text-yellow-500")}>
                                    Δ {percent}%
                                </span>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default ComparisonTable;