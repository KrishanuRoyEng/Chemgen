export const PROPERTY_TOOLTIPS = {
    // Primary Properties
    mw: "Molecular Weight (Da): Determines absorption and permeability. Optimal drug-like range: < 500 Da.",
    logp: "LogP (Lipophilicity): Measures solubility in lipids vs water. Critical for passing through cell membranes. Optimal: 0-5.",

    // Neural Constraints
    tpsa: "Topological Polar Surface Area (Å²): Sum of polar atoms surfaces. Affects drug transport. < 140 Å² is good for cell permeability.",
    hbd: "H-Bond Donors: Number of hydrogen bond donors (e.g., OH, NH). Too many (>5) reduce permeability.",
    hba: "H-Bond Acceptors: Number of hydrogen bond acceptors (e.g., N, O). Too many (>10) reduce permeability.",
    rings: "Ring Count: Number of aromatic/non-aromatic rings. Affects molecular rigidity and target binding.",
    qed: "Quantitative Estimation of Drug-likeness: A score from 0 (non-drug) to 1 (ideal drug) based on multiple properties.",
    toxicity: "Predicted Toxicity: Neural network probability of toxicity. Lower values are safer.",
    rot: "Rotatable Bonds: Number of non-rigid bonds. Too flexible (>10) molecules bind poorly to targets.",

    // Domain Specific
    adhesion: "Adhesion Strength (Materials): Predicted capability to adhere to surfaces. Higher is stronger.",
    affinity: "Binding Affinity (-logKd): Strength of interaction with a biological target. Higher values mean tighter binding.",

    // Computed / Analysis
    sas: "Synthetic Accessibility Score: 1 (Easy) to 10 (Hard). Estimates the difficulty of synthesizing the molecule."
};
