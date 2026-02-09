/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./src/**/*.{js,jsx,ts,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'cyber-black': '#0a0a0a',
                'cyber-slate': '#1e1e1e',
                'neon-green': '#00ff9d',
                'neon-blue': '#00d0ff',
                'neon-purple': '#bf00ff',
                'holo-blue': 'rgba(0, 208, 255, 0.1)',
                'holo-purple': 'rgba(191, 0, 255, 0.1)',
                'danger-red': '#ff2a6d',
            },
            fontFamily: {
                mono: ['"Fira Code"', 'monospace'], // Suggest installing a monospace font or using system default
                sans: ['Inter', 'sans-serif'],
            },
            boxShadow: {
                'neon-green': '0 0 10px #00ff9d, 0 0 20px #00ff9d',
                'neon-blue': '0 0 10px #00d0ff, 0 0 20px #00d0ff',
                'neon-purple': '0 0 10px #bf00ff, 0 0 20px #bf00ff',
            }
        },
    },
    plugins: [],
}
