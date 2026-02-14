import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';

const Tooltip = ({ content, children }) => {
    const [isVisible, setIsVisible] = useState(false);
    const triggerRef = useRef(null);
    const [coords, setCoords] = useState({ left: 0, top: 0 });
    const [placement, setPlacement] = useState('top');

    // Define the gap between the element and the tooltip arrow (10px buffer)
    const GAP = 10;

    const updatePosition = () => {
        if (triggerRef.current) {
            const rect = triggerRef.current.getBoundingClientRect();
            const tooltipHeightEstimate = 100; // Buffer for logic

            // Check if there is enough space above
            const spaceAbove = rect.top;

            let newPlacement = 'top';
            let calculatedTop = rect.top - GAP; // Default: Shift UP by Gap

            // Flip to bottom if crowded (e.g. at top of screen)
            if (spaceAbove < tooltipHeightEstimate) {
                newPlacement = 'bottom';
                calculatedTop = rect.bottom + GAP; // Shift DOWN by Gap
            }

            setCoords({
                left: rect.left + rect.width / 2, // Center horizontally
                top: calculatedTop,
            });
            setPlacement(newPlacement);
        }
    };

    useEffect(() => {
        if (isVisible) {
            updatePosition();
            window.addEventListener('resize', updatePosition);
            window.addEventListener('scroll', updatePosition);
        }
        return () => {
            window.removeEventListener('resize', updatePosition);
            window.removeEventListener('scroll', updatePosition);
        };
    }, [isVisible]);

    return (
        <>
            {/* Trigger Wrapper */}
            <div
                ref={triggerRef}
                className="relative inline-block cursor-help"
                onMouseEnter={() => {
                    updatePosition();
                    setIsVisible(true);
                }}
                onMouseLeave={() => setIsVisible(false)}
            >
                {children}
            </div>

            {/* Portal Tooltip */}
            {createPortal(
                <AnimatePresence>
                    {isVisible && (
                        <motion.div
                            initial={{ opacity: 0, y: placement === 'top' ? 5 : -5, scale: 0.95 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            transition={{ duration: 0.15, ease: "easeOut" }}
                            style={{
                                left: coords.left,
                                top: coords.top,
                                position: 'fixed',
                                // Transform Origin logic ensures it grows OUT from the element
                                transform: placement === 'top' ? 'translate(-50%, -100%)' : 'translate(-50%, 0)',
                                zIndex: 9999,
                                pointerEvents: 'none',
                            }}
                            className="w-64 p-3 bg-black/90 border border-neon-blue/30 rounded-lg shadow-[0_0_15px_rgba(0,163,255,0.2)] backdrop-blur-md"
                        >
                            {/* FIXED ARROW LOGIC */}
                            <div className={clsx(
                                "absolute left-1/2 -translate-x-1/2 w-2 h-2 bg-black/90 rotate-45 border-neon-blue/30",
                                placement === 'top'
                                    ? "-bottom-1 border-r border-b"
                                    : "-top-1 border-l border-t"
                            )}></div>

                            <p className="text-[10px] text-gray-300 font-mono leading-relaxed relative z-10">
                                {content}
                            </p>
                        </motion.div>
                    )}
                </AnimatePresence>,
                document.body
            )}
        </>
    );
};

export default Tooltip;