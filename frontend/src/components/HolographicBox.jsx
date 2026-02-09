import React, { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as $3Dmol from '3dmol';

const HolographicBox = ({ loading, smiles, result }) => {
    const viewerRef = useRef(null);
    const containerRef = useRef(null);

    useEffect(() => {
        if (!containerRef.current) return;

        // Initialize viewer
        // Use a transparent background for that "hologram" feel if possible, 
        // but 3Dmol often needs a color. Let's try deep grey.
        const viewer = $3Dmol.createViewer(containerRef.current, { backgroundColor: '#1e1e1e' }); // cyber-slate
        viewerRef.current = viewer;

        return () => {
            // Cleanup if needed
        };
    }, []);

    useEffect(() => {
        const viewer = viewerRef.current;
        if (!viewer) return;

        if (loading) {
            viewer.clear();
            viewer.render();
            return;
        }

        if (smiles) {
            viewer.clear();
            if (result?.mol_block) {
                // If backend sent 3D coordinates, use them (Format: 'sdf')
                viewer.addModel(result.mol_block, "sdf");
            } else if (smiles) {
                // Fallback to SMILES if 3D failed (Format: 'smi')
                viewer.addModel(smiles, "smi");
            }
            viewer.setStyle({}, { stick: { radius: 0.2, colorscheme: 'cyanCarbon' }, sphere: { scale: 0.3 } }); // CyanCarbon looks sci-fi
            viewer.zoomTo();
            viewer.render();
        }
    }, [loading, smiles, result]); // Re-run when smiles changes

    return (
        <div className="relative w-full h-[500px] glass-panel border-2 border-neon-blue/30 overflow-hidden flex items-center justify-center">
            {/* Background Grid Effect */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(0,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(0,255,255,0.05)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none"></div>

            {/* Loading State: Chaotic Sphere */}
            <AnimatePresence>
                {loading && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.5 }}
                        animate={{ opacity: 1, scale: 1, rotate: 360 }}
                        exit={{ opacity: 0, scale: 1.5, filter: "blur(10px)" }}
                        transition={{ duration: 0.5, repeat: Infinity, ease: "linear" }}
                        className="absolute inset-0 flex items-center justify-center z-20 pointer-events-none"
                    >
                        <div className="w-32 h-32 rounded-full border-4 border-t-neon-blue border-r-transparent border-b-neon-purple border-l-transparent animate-spin blur-sm"></div>
                        <div className="absolute w-24 h-24 rounded-full border-2 border-white/50 animate-ping"></div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Flash Effect on Success */}
            <AnimatePresence>
                {!loading && smiles && (
                    <motion.div
                        initial={{ opacity: 1 }}
                        animate={{ opacity: 0 }}
                        transition={{ duration: 0.5 }}
                        className="absolute inset-0 bg-white z-30 pointer-events-none"
                    />
                )}
            </AnimatePresence>

            {/* 3D Mol Container */}
            <div ref={containerRef} className="w-full h-full z-10 relative" id="viewer_3d_container"></div>

            {/* Corner Accents */}
            <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-neon-blue z-20"></div>
            <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-neon-blue z-20"></div>
            <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-neon-blue z-20"></div>
            <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-neon-blue z-20"></div>

            {/* RAW DATA TERMINAL (So you can see the SMILES) */}
            {!loading && smiles && (
                <div className="absolute bottom-0 left-0 right-0 bg-black/80 backdrop-blur border-t border-neon-blue/30 p-3 font-mono text-[10px] text-neon-blue/80 overflow-hidden whitespace-nowrap z-30">
                    <div className="flex justify-between items-center mb-1 opacity-50">
                        <span>RAW_OUTPUT_STREAM</span>
                        <span>LEN: {smiles.length}</span>
                    </div>
                    <div className="animate-pulse text-white select-all">
                        {">"} {smiles}
                    </div>
                </div>
            )}
        </div>
    );
};

export default HolographicBox;
