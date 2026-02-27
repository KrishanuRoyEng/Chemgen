import React from 'react';
import clsx from 'clsx';
import Tooltip from './Tooltip';
import { PROPERTY_TOOLTIPS } from '../constants/tooltips';

// 1. Added a default empty object to targets to prevent undefined crashes
const ComparisonTable = ({ targets = {}, result, domain }) => {
    // 2. Added !targets to the safety return
    if (!result || !result.properties || !targets) return null;
    
    const actual = result.properties;

    // Define all metrics and which domains they belong to
    // Added optional chaining (?.) just in case a specific target isn't loaded yet
    const allMetrics = [
        { id: 'mw', label: 'Weight', target: targets?.mw, actual: actual.mw, unit: 'Da', domains: ['drug', 'material', 'biomolecule'] },
        { id: 'logp', label: 'LogP', target: targets?.logp, actual: actual.logp, unit: '', domains: ['drug', 'material', 'biomolecule'] },
        { id: 'tpsa', label: 'TPSA', target: targets?.tpsa, actual: actual.tpsa, unit: 'Å²', domains: ['drug', 'material', 'biomolecule'] },
        { id: 'qed', label: 'QED', target: targets?.qed, actual: actual.qed, unit: '', domains: ['drug'] },
        { id: 'hbd', label: 'H-Donors', target: targets?.hbd, actual: actual.hbd, unit: '', domains: ['drug', 'biomolecule'] },
        { id: 'hba', label: 'H-Acceptors', target: targets?.hba, actual: actual.hba, unit: '', domains: ['drug', 'biomolecule'] },
        { id: 'rings', label: 'Rings', target: targets?.rings, actual: actual.rings, unit: '', domains: ['drug', 'material', 'biomolecule'] },
        { id: 'rot', label: 'Flexibility (Rot)', target: targets?.rot, actual: actual.rot, unit: '', domains: ['material', 'drug'] },
        { id: 'adhesion', label: 'Adhesion Str.', target: targets?.adhesion, actual: actual.adhesion, unit: '/ 10', domains: ['material'] },
        { id: 'affinity', label: 'Binding Affinity', target: targets?.affinity, actual: actual.affinity, unit: '/ 10', domains: ['biomolecule'] },
    ];

    // Filter metrics based on the current domain
    const visibleMetrics = allMetrics.filter(m => m.domains.includes(domain));

    return (
        <div className="mt-4 border-t border-white/10 pt-4">
            <h3 className="text-[10px] font-mono text-gray-500 mb-2 uppercase tracking-widest">Convergence Report</h3>
            <div className="space-y-1">
                {visibleMetrics.map((m) => {
                    // Safe Percentage Calculation (prevents division by zero or NaN)
                    const targetVal = Number(m.target) || 0;
                    const actualVal = Number(m.actual) || 0;
                    let percent = 0;
                    
                    if (targetVal === 0) {
                        percent = actualVal === 0 ? 0 : 100;
                    } else {
                        const diff = Math.abs(targetVal - actualVal);
                        percent = Math.min((diff / targetVal) * 100, 999);
                    }

                    return (
                        <div key={m.id} className="flex justify-between items-center text-[10px] font-mono">
                            <Tooltip content={PROPERTY_TOOLTIPS?.[m.id] || "Target vs Generated property comparison."}>
                                <span className="text-gray-400 cursor-help border-b border-dotted border-gray-600 hover:text-gray-200 transition-colors">
                                    {m.label}
                                </span>
                            </Tooltip>
                            <div className="flex gap-3 text-right">
                                <span className="text-gray-600 w-16">Tar: {targetVal}</span>
                                <span className="text-white w-20">Gen: {actualVal} {m.unit}</span>
                                <span className={clsx(
                                    "w-14 text-right",
                                    percent < 15 ? "text-neon-green" : percent < 40 ? "text-yellow-500" : "text-red-500"
                                )}>
                                    Δ {percent.toFixed(1)}%
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