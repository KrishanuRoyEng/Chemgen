import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';

const Tooltip = ({ content, children }) => {
    const [isVisible, setIsVisible] = useState(false);
    const triggerRef = useRef(null);
    const [coords, setCoords] = useState({ left: 0, top: 0 });
    const [placement, setPlacement] = useState('top'); // 'top' or 'bottom'

    const updatePosition = () => {
        if (triggerRef.current) {
            const rect = triggerRef.current.getBoundingClientRect();
            const spaceAbove = rect.top;
            const tooltipHeightEstimate = 100; // Approximate hidden height

            let newPlacement = 'top';
            let top = rect.top;

            if (spaceAbove < tooltipHeightEstimate) {
                newPlacement = 'bottom';
                top = rect.bottom;
            }

            setCoords({
                left: rect.left + rect.width / 2,
                top: top,
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
            <div
                ref={triggerRef}
                className="relative flex items-center inline-block"
                onMouseEnter={() => {
                    updatePosition();
                    setIsVisible(true);
                }}
                onMouseLeave={() => setIsVisible(false)}
            >
                {children}
            </div>
            {createPortal(
                <AnimatePresence>
                    {isVisible && (
                        <motion.div
                            initial={{ opacity: 0, y: placement === 'top' ? 10 : -10, scale: 0.95 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, y: placement === 'top' ? 10 : -10, scale: 0.95 }}
                            transition={{ duration: 0.15 }}
                            style={{
                                left: coords.left,
                                top: coords.top,
                                position: 'fixed',
                                transform: placement === 'top' ? 'translate(-50%, -100%)' : 'translate(-50%, 0)',
                                zIndex: 9999,
                                pointerEvents: 'none',
                                marginTop: placement === 'bottom' ? '8px' : '0',
                                marginBottom: placement === 'top' ? '8px' : '0'
                            }}
                            className="w-64 p-3 bg-black/90 border border-neon-blue/30 rounded-lg shadow-[0_0_15px_rgba(0,163,255,0.2)] backdrop-blur-sm"
                        >
                            {/* Arrow Logic */}
                            <div className={clsx(
                                "absolute left-1/2 -translate-x-1/2 w-2 h-2 bg-black/90 border-r border-b border-neon-blue/30 rotate-45",
                                placement === 'top' ? "-bottom-1" : "-top-1 rotate-[225deg]"
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
