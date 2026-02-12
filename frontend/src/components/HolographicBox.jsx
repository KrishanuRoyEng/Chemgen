import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as $3Dmol from '3dmol';
import { RotateCw, RefreshCw, Layers, Info } from 'lucide-react';
import clsx from 'clsx';

const HolographicBox = ({ processStatus, smiles, result }) => {
    const viewerRef = useRef(null);
    const containerRef = useRef(null);
    const [autoSpin, setAutoSpin] = useState(false);
    const rotationIntervalRef = useRef(null);
    const [hoveredAtom, setHoveredAtom] = useState(null);

    // Derived loading state
    const loading = processStatus === 'SEEDING' || processStatus === 'SYNTHESIZING';

    useEffect(() => {
        if (!containerRef.current) return;

        // Initialize viewer
        const viewer = $3Dmol.createViewer(containerRef.current, { backgroundColor: '#1e1e1e' }); // cyber-slate
        viewerRef.current = viewer;

        // Add Zoom/Interaction Constraints via Canvas Events
        const canvas = containerRef.current.querySelector('canvas');
        if (canvas) {
            canvas.addEventListener('wheel', handleZoom, { passive: false });
        }

        return () => {
            if (canvas) canvas.removeEventListener('wheel', handleZoom);
            if (rotationIntervalRef.current) clearInterval(rotationIntervalRef.current);
        };
    }, []);

    // Custom Zoom Handler to prevent "disappearing" molecule
    const handleZoom = (e) => {
        // We allow default zoom but could clamp here if needed.
        // For now, we rely on the standard behavior but this hook exists for future clamping.
    };

    // Handle Auto-Spin
    useEffect(() => {
        if (autoSpin && viewerRef.current) {
            const interval = setInterval(() => {
                viewerRef.current.rotate(1);
            }, 50); // Spin 1 degree every 50ms
            rotationIntervalRef.current = interval;
        } else {
            if (rotationIntervalRef.current) clearInterval(rotationIntervalRef.current);
            rotationIntervalRef.current = null;
        }

        return () => {
            if (rotationIntervalRef.current) clearInterval(rotationIntervalRef.current);
        };
    }, [autoSpin]);

    useEffect(() => {
        const viewer = viewerRef.current;
        if (!viewer) return;

        if (loading) {
            viewer.clear();
            viewer.render();
            setHoveredAtom(null);
            return;
        }

        if (result?.mol_block || smiles) {
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

            // ATOM HOVER INTERACTION
            viewer.setHoverable({}, true, (atom, viewer, event, container) => {
                if (atom && !loading) {
                    setHoveredAtom({
                        element: atom.elem,
                        index: atom.serial !== undefined ? atom.serial : atom.index,
                        charge: atom.charge,
                        x: event.clientX,
                        y: event.clientY
                    });
                }
            }, (atom) => {
                setHoveredAtom(null);
            });

            viewer.render();
        }
    }, [loading, smiles, result]);

    return (
        <div className="relative w-full min-h-[350px] md:h-[500px] glass-panel border-2 border-neon-blue/30 overflow-hidden flex items-center justify-center group">
            {/* Background Grid Effect */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(0,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(0,255,255,0.05)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none"></div>

            {/* SEEDED STATE OVERLAY */}
            <AnimatePresence>
                {processStatus === 'SEEDED' && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 flex flex-col items-center justify-center z-30 bg-black/60 backdrop-blur-sm"
                    >
                        <Layers className="text-neon-green w-16 h-16 mb-4 animate-bounce" />
                        <h3 className="text-xl font-bold text-neon-green tracking-widest uppercase">Data Seeded</h3>
                        <p className="text-sm font-mono text-gray-400 mt-2">Ready for Synthesis</p>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Loading State: Chaotic Sphere */}
            <AnimatePresence mode="wait">
                {loading && (
                    <motion.div
                        key="loader"
                        initial={{ opacity: 0, scale: 0.5 }}
                        animate={{ opacity: 1, scale: 1, rotate: 360 }}
                        exit={{ opacity: 0, scale: 1.5, filter: "blur(10px)" }}
                        transition={{ duration: 0.5, repeat: Infinity, ease: "linear" }}
                        className="absolute inset-0 flex items-center justify-center z-20 pointer-events-none"
                    >
                        <div className="w-32 h-32 rounded-full border-4 border-t-neon-blue border-r-transparent border-b-neon-purple border-l-transparent animate-spin blur-sm"></div>
                        <div className="absolute w-24 h-24 rounded-full border-2 border-white/50 animate-ping"></div>
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-[10px] font-mono whitespace-nowrap text-white/50 animate-pulse">
                            {processStatus === 'SEEDING' ? "INITIALIZING SEED..." : "SYNTHESIZING..."}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ATOM DETAILS TOOLTIP (Using Portal behavior visually but absolute here for simplicity relative to canvas) */}
            {hoveredAtom && (
                <div
                    className="fixed z-50 pointer-events-none p-2 bg-black/90 border border-neon-blue rounded shadow-lg backdrop-blur text-[10px] font-mono text-white"
                    style={{ left: hoveredAtom.x + 15, top: hoveredAtom.y + 15 }}
                >
                    <div className="flex items-center gap-1 mb-1 border-b border-white/20 pb-1">
                        <Info size={10} className="text-neon-blue" />
                        <span className="font-bold text-neon-blue">ATOM DETAILS</span>
                    </div>
                    <div>ELEM: <span className="text-yellow-400">{hoveredAtom.element}</span></div>
                    <div>IDX:  <span className="text-gray-400">{hoveredAtom.index}</span></div>
                    {hoveredAtom.charge !== undefined && <div>CHG:  <span className="text-red-400">{hoveredAtom.charge}</span></div>}
                </div>
            )}

            {/* Flash Effect on Success */}
            <AnimatePresence>
                {!loading && smiles && processStatus === 'COMPLETE' && (
                    <motion.div
                        initial={{ opacity: 1 }}
                        animate={{ opacity: 0 }}
                        transition={{ duration: 0.5 }}
                        className="absolute inset-0 bg-white z-30 pointer-events-none"
                    />
                )}
            </AnimatePresence>

            {/* 3D Mol Container */}
            <div ref={containerRef} className="w-full h-full z-10 relative cursor-move" id="viewer_3d_container"></div>

            {/* CONTROLS OVERLAY */}
            <div className="absolute top-4 right-4 z-40 flex flex-col gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <button
                    onClick={() => {
                        if (viewerRef.current) {
                            viewerRef.current.zoomTo();
                            viewerRef.current.render();
                        }
                    }}
                    className="p-2 bg-black/50 border border-white/20 rounded hover:bg-neon-blue/20 hover:border-neon-blue/50 text-white transition-all"
                    title="Reset View"
                >
                    <RefreshCw size={16} />
                </button>
                <button
                    onClick={() => setAutoSpin(!autoSpin)}
                    className={clsx(
                        "p-2 border rounded transition-all text-white",
                        autoSpin ? "bg-neon-blue/20 border-neon-blue text-neon-blue animate-pulse" : "bg-black/50 border-white/20 hover:bg-neon-blue/20 hover:border-neon-blue/50"
                    )}
                    title="Toggle Auto-Spin"
                >
                    <RotateCw size={16} className={clsx(autoSpin && "animate-spin")} />
                </button>
            </div>

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
