# Novel Research Plan: AI/ML-Driven Inverse Design of Electromagnetic Resonators

## 1) Proposed Paper Direction (Novel by Design)

### Working title
Physics-Informed, Data-Efficient AI/ML Framework for Inverse Design of All-Metal Electromagnetic Resonators

### Core novelty claims
1. Frequency-conditioned inverse generation for resonator geometry:
Use a conditional generative model (cVAE or diffusion-style tabular generator) that maps target frequency and constraints directly to geometry parameters (r1, r2, r3, r4, t, d1, d2, h, p).

2. Physics-informed AI/ML training (not only data fitting):
Embed electromagnetic priors and geometric constraints directly in the loss function, including manufacturability and monotonic/ordering rules (for example r1 > r2 > r3 > r4, feasible ranges, physically valid periodicity).

3. Active-learning loop for minimum simulation budget:
Train a surrogate model, quantify uncertainty, and iteratively request only the most informative new CST/HFSS simulations.
Target: significantly fewer full-wave simulations than conventional brute-force dataset expansion.

4. Explainable AI/ML for design insight:
Use SHAP or permutation sensitivity and local gradient analysis to discover which geometric parameters dominate frequency shift and resonance quality in different bands.

This combination (generative inverse model + physics-informed constraints + uncertainty-driven active learning + explainability) is the main novelty axis.

---

## 2) What Is Already Available (From Your Files)

### From paper1and2.txt
- Prior work basis includes all-metal metamaterial resonator-inspired structures and strong emphasis on compactness and efficiency.
- The review material confirms standard ANN workflow: full-wave simulation data -> train surrogate -> inverse/forward prediction.
- It also identifies existing gaps that your paper can attack directly:
  - Data dependence and expensive simulation generation.
  - Generalization problems for unseen parameter regions.
  - Need for semi-supervised/physics-guided methods.

### From your XLSX data snapshot
- Structured parameter map from 1.0 GHz to 12.0 GHz, with paired design records across the band.
- The data now has two linked geometry layers:
  - Full-structure dimensions: waveguide coupler length (L), inner radius of waveguide coupler (r), circular waveguide length (P), inner radius of CeSRR unit cell (R), separation between CeSRRs (p), and total MTM SWS length (D).
  - Unit-cell dimensions: outer radius of outer circumference (r1), inner radius of outer circumference (r2), outer radius of inner circumference (r3), inner radius of inner circumference (r4), thickness (t), groove width (d1), metal bridge (d2), groove height (h), and periodicity (p).
- This makes the dataset suitable for both system-level inverse design and unit-cell-conditioned geometry generation.
- It is an excellent seed for supervised inverse mapping, constrained generative modeling, and frequency-conditioned active learning.

---

## 3) Recent AI/ML Trends To Build On (Internet Scan, 2023+)

Representative directions and papers found during search:

1. Deep-learning inverse metasurface design with strong performance in absorber/polarization tasks.
- DOI: 10.1364/OL.518786
- Title: Deep learning-based inverse design of multi-functional metasurface absorbers (2024)

2. Small-data inverse design strategy with statistically filtered training data before expensive simulation.
- DOI: 10.1002/MOP.34068
- Title: Inverse design of reflective metasurface antennas using deep learning from small-scale statistically random pico-cells (2024)

3. PINN methods entering electromagnetics analysis workflows.
- DOI: 10.13052/2023.ACES.J.381102
- Title: Physics-informed Neural Networks for the Resolution of Analysis Problems in Electromagnetics (2024)

4. Hybrid numerical + PINN frameworks for EM PDE solving.
- DOI: 10.1109/ACCESS.2024.3500039
- Title: A Novel Hybrid Boundary Element-Physics Informed Neural Network Method for Numerical Solutions in Electromagnetics (2024)

5. PINN-based waveguide eigenanalysis as a practical EM application direction.
- DOI: 10.1109/ACCESS.2024.3452160
- Title: A Physics-Informed Neural Network-Based Waveguide Eigenanalysis (2024)

6. Geometry representation evolution toward mesh-free 3D deep learning EM inversion.
- DOI: 10.1109/TMTT.2023.3248174
- Title: A Mesh-Free 3-D Deep Learning Electromagnetic Inversion Method Based on Point Clouds (2023)

Research trend takeaway:
Recent work is moving from plain ANN fitting to constrained, physics-guided, and data-efficient inverse design. Your paper should position exactly there.

---

## 4) Research Questions (RQ) and Hypotheses

### RQ1
Can an AI/ML inverse model generate valid resonator geometries from target frequency with lower error than conventional regression baselines?

### RQ2
Does physics-informed training improve extrapolation and physical validity versus purely data-driven models?

### RQ3
How much can uncertainty-guided active learning reduce the number of full-wave simulations while maintaining target accuracy?

### Hypotheses
- H1: Physics-informed models will reduce invalid geometry predictions and improve out-of-distribution stability.
- H2: Active learning will cut simulation budget by at least 30-50% for same error target.
- H3: A generative inverse model will provide multiple feasible geometry candidates per target frequency (better design flexibility than one-to-one regression).

---

## 5) Methodology Plan (End-to-End)

### Phase A: Data engineering and quality control
1. Normalize frequency strings (for example "1 GHz", "1.2GHz", "10.2 GHz") into numeric GHz values.
2. Parse and clean all parameter rows; flag and fix anomalies (for example malformed numeric text like "0..25").
3. Enforce dimensional constraints and remove physically impossible records.
4. Align the full-structure and unit-cell tables by frequency and create one unified schema for modeling.
5. Build versioned datasets:
- D0: raw cleaned table
- D1: physics-validated subset
- D2: augmented pool (after targeted simulation)

### Phase B: Baseline models
1. Forward model: geometry -> frequency response proxies
- Algorithms: XGBoost, Random Forest, MLP
2. Inverse model: target frequency -> geometry
- Algorithms: MLP, mixture density network (for one-to-many mapping)

### Phase C: Novel model stack
1. Physics-informed surrogate:
- Neural model with penalty terms for geometry ordering, smoothness, and feasible EM behavior.
2. Generative inverse module:
- Conditional VAE (first choice) or diffusion-style tabular generator to output multiple valid geometry candidates.
3. Uncertainty estimator:
- Deep ensemble or MC dropout to estimate confidence for each candidate.

### Phase D: Active-learning simulation loop
1. Select high-uncertainty/high-impact candidates.
2. Run full-wave EM simulation only for selected candidates.
3. Append new samples and retrain surrogate/inverse model.
4. Repeat until error and validity targets converge.

### Phase E: Explainability and design rules
1. Global sensitivity by SHAP/permutation.
2. Local sensitivity around selected target frequencies.
3. Convert findings into practical design rules for resonator synthesis.

---

## 6) Experimental Protocol

### Data split strategy
- Interpolation split: random train/val/test.
- Extrapolation split: hold out full frequency bands (hard test).

### Metrics
1. Frequency prediction MAE and RMSE.
2. Geometry feasibility rate (percentage of physically valid outputs).
3. Simulation agreement score (predicted vs full-wave verified response).
4. Active-learning efficiency:
- Error versus number of full-wave simulations.

### Ablation studies
1. Without physics-informed loss.
2. Without active learning.
3. Deterministic inverse regression vs generative inverse approach.

---

## 7) Expected Contributions (Publishable)

1. A data-efficient AI/ML inverse design framework for electromagnetic resonators.
2. A validated physics-informed training strategy that improves robustness.
3. Quantified simulation-budget reduction through active learning.
4. Interpretable geometry-frequency design rules for practical resonator engineering.

---

## 8) Suggested Paper Structure

1. Introduction and motivation
2. Related work (AI/ML inverse design, PINN in electromagnetics, active learning)
3. Resonator geometry and dataset description
4. Proposed AI/ML framework
5. Experiments and ablations
6. Discussion (validity, limits, scalability)
7. Conclusion and future work

---

## 9) Timeline (16 Weeks)

1. Weeks 1-2: Data cleaning, parsing, and quality checks.
2. Weeks 3-4: Baseline forward/inverse models.
3. Weeks 5-8: Physics-informed surrogate + generative inverse model.
4. Weeks 9-11: Active-learning simulation loop.
5. Weeks 12-13: Explainability and design-rule extraction.
6. Weeks 14-15: Writing, figures, and comparison tables.
7. Week 16: Final revision and submission package.

---

## 10) Immediate Next Actions

1. Finalize clean machine-readable dataset from the XLSX sheet, including both full-structure and unit-cell tables.
2. Define parameter bounds and hard physical constraints for the two geometry layers.
3. Implement baseline inverse regression and report first benchmark.
4. Start physics-informed + generative inverse prototype.

This plan is intentionally structured to produce a novel paper rather than a routine ANN fitting study.
