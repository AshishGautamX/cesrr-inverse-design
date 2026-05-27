# Research Plan: First Physics-Informed, Data-Efficient, Interpretable Inverse Design Framework for All-Metal Metamaterial-Inspired Slow Wave Structures

## 1) Proposed Paper Direction — Precisely Framed Novelty

### Working Title
**"Physics-Informed, Data-Efficient, and Interpretable Inverse Design of All-Metal CeSRR-Based Slow Wave Structures for Vacuum Electron Devices"**

### Why This Framing — Novelty Audit

#### What already exists (prior art — NOT claimed as novel)
| Component | Prior Art |
|---|---|
| DNN / cVAE for inverse design of SWS | IEEE 2022 — already done |
| Surrogate + EM simulation loop for SWS optimization | IEEE 2024 — already done |
| Physics-informed EM constraints | Exists in optical/metasurface domain |
| SHAP for EM antenna design | Scientific Reports, July 2025 |
| Active learning for EM simulation reduction | Exists in antenna/microwave domain |

#### What is genuinely novel — the gap nobody has filled
The individual ingredients above are known. However, applied **together** to **all-metal CeSRR-based slow wave structures for vacuum electron devices (TWT/HPM)**, no such work exists. Specifically:

1. **Physics-informed geometry ordering constraints (r1 > r2 > r3 > r4) embedded in the loss function** for this specific CeSRR unit-cell topology — **zero prior work found.**
2. **SHAP-derived design rules for CeSRR/all-metal SWS geometry-to-frequency relationships** — **zero prior work found.**
3. **Uncertainty-guided active learning to reduce CST/HFSS simulation budget for this device class** — **zero prior work found.**

#### The publishable claim (one sentence)
> *We present the first physics-informed, data-efficient, and interpretable inverse design framework for all-metal CeSRR-based metamaterial slow wave structures used in high-power microwave and traveling-wave tube devices.*

The novelty is **not** a new ML method. It is the **first ML treatment of this specific device class** using a combination that closes three simultaneously open sub-gaps in the domain.

---

## 2) What Is Already Available (From Your Files)

### From paper1and2.txt
- Prior work confirms standard ANN workflow: full-wave simulation data → train surrogate → inverse/forward prediction.
- Existing gaps directly attackable:
  - Expensive full-wave simulation dependency.
  - Poor generalization to unseen parameter regions.
  - Absence of physics-guided or semi-supervised methods for this device class.
- **Paper 1 (IEEE 2022)**: DNN/cVAE for SWS inverse design — establishes the baseline this paper must outperform or extend.
- **Paper 2 (IEEE 2024)**: Surrogate + EM loop for SWS optimization — establishes the simulation-efficiency baseline.

### From your XLSX data snapshot
- Structured parameter map from 1.0 GHz to 12.0 GHz, with paired design records across the band.
- Two linked geometry layers in the dataset:
  - **Full-structure dimensions**: waveguide coupler length (L), inner radius of waveguide coupler (r), circular waveguide length (P), inner radius of CeSRR unit cell (R), separation between CeSRRs (p), total MTM SWS length (D).
  - **Unit-cell dimensions**: outer radius of outer circumference (r1), inner radius of outer circumference (r2), outer radius of inner circumference (r3), inner radius of inner circumference (r4), thickness (t), groove width (d1), metal bridge (d2), groove height (h), periodicity (p).
- The **structural constraint r1 > r2 > r3 > r4** is a hard physical law of the CeSRR topology — embedding this in the loss function is the first such formulation for this device class.
- Dataset is suitable for: supervised inverse mapping, constrained generative modeling, frequency-conditioned active learning, and SHAP-based design rule extraction.

---

## 3) Literature Context and Positioning (2023+)

### What the field has (general domain)

1. Deep-learning inverse metasurface design — absorber/polarization tasks.
   - DOI: 10.1364/OL.518786 (2024)

2. Small-data inverse design with statistically filtered training for metasurface antennas.
   - DOI: 10.1002/MOP.34068 (2024)

3. PINN methods entering EM analysis workflows.
   - DOI: 10.13052/2023.ACES.J.381102 (2024)

4. Hybrid boundary element + PINN for EM PDE solving.
   - DOI: 10.1109/ACCESS.2024.3500039 (2024)

5. PINN-based waveguide eigenanalysis.
   - DOI: 10.1109/ACCESS.2024.3452160 (2024)

6. Mesh-free 3D deep learning EM inversion (point clouds).
   - DOI: 10.1109/TMTT.2023.3248174 (2023)

7. SHAP for EM antenna design.
   - Scientific Reports (July 2025)

### What the field does NOT have (the gap)
- None of the above work targets **all-metal CeSRR-based SWS** for vacuum electron devices (TWT / HPM).
- No prior work combines physics-informed ordering constraints + SHAP design rules + uncertainty-guided active learning **in this device class**.
- The HPM/TWT domain has remained outside the reach of physics-informed generative inverse design frameworks.

### Positioning statement for the introduction
> *While physics-informed neural networks, generative inverse models, and active learning have each been applied to metasurface and antenna structures, no prior work has applied this combined methodology to all-metal metamaterial-inspired slow wave structures for vacuum electron devices. This paper addresses that gap.*

---

## 4) Research Questions and Hypotheses

### RQ1 — Inverse Design Quality
Can a physics-constrained generative model produce valid CeSRR unit-cell geometries from a target frequency with higher feasibility rate and lower prediction error than conventional regression baselines?

### RQ2 — Physics-Informed Gain
Does embedding the CeSRR-specific geometric ordering constraint (r1 > r2 > r3 > r4) and EM priors in the loss function improve physical validity and out-of-distribution generalization compared to purely data-driven models?

### RQ3 — Simulation Budget Reduction
By how much does uncertainty-guided active learning reduce the number of required CST/HFSS full-wave simulations while maintaining target accuracy for this device class?

### RQ4 — Interpretable Design Rules
What geometry-frequency relationships for CeSRR-based SWS can be discovered through SHAP analysis, and are they consistent with known EM physics?

### Hypotheses
- **H1**: Physics-informed models will reduce invalid geometry predictions and improve extrapolation stability compared to unconstrained models.
- **H2**: Active learning will cut simulation budget by at least 30–50% for the same error target.
- **H3**: A generative inverse model will provide multiple feasible geometry candidates per target frequency (better design flexibility than one-to-one regression).
- **H4**: SHAP will identify r1 and periodicity (p) as dominant parameters for frequency control, consistent with CeSRR EM theory — confirming model physical plausibility.

---

## 5) Methodology Plan (End-to-End)

### Phase A: Data Engineering and Quality Control
1. Normalize frequency strings (e.g., "1 GHz", "1.2GHz", "10.2 GHz") into numeric GHz values.
2. Parse and clean all parameter rows; flag and fix anomalies (e.g., malformed numeric text like "0..25").
3. Enforce hard dimensional constraints — flag and remove physically impossible records.
4. **Validate the r1 > r2 > r3 > r4 ordering** across all unit-cell records; quantify violations for paper reporting.
5. Align full-structure and unit-cell tables by frequency into one unified schema.
6. Build versioned datasets:
   - D0: raw cleaned table
   - D1: physics-validated subset (ordering constraints enforced)
   - D2: augmented pool (after targeted simulation via active learning)

### Phase B: Baseline Models (Reference Points for Claims)
1. **Forward model**: geometry → frequency response proxies
   - Algorithms: XGBoost, Random Forest, MLP
2. **Inverse model**: target frequency → geometry
   - Algorithms: MLP, mixture density network (one-to-many mapping)
3. These baselines establish the floor that the novel framework must beat.

### Phase C: Novel Model Stack (The Contribution)

#### C1 — Physics-Informed Surrogate
- Neural surrogate with penalty terms in the loss function for:
  - **Geometry ordering**: r1 > r2 > r3 > r4 (CeSRR-specific, first formulation for this topology)
  - Smoothness and feasible EM behavior across frequency band
  - Manufacturability bounds on t, d1, d2, h, p

#### C2 — Generative Inverse Module
- **Conditional VAE (cVAE)** conditioned on target frequency: outputs multiple valid geometry candidates per input frequency.
- The physics-informed penalty from C1 is carried into the cVAE decoder loss.

#### C3 — Uncertainty Estimator
- Deep ensemble or MC dropout to estimate confidence per candidate geometry.
- Confidence scores drive active learning selection.

### Phase D: Uncertainty-Guided Active Learning Loop (First for HPM/TWT SWS)
1. Select high-uncertainty / high-impact geometry candidates from cVAE output.
2. Run full-wave EM simulation (CST/HFSS) **only** for selected candidates.
3. Append new samples to D2 and retrain surrogate + inverse model.
4. Repeat until error and validity targets converge.
5. Report: error vs. number of full-wave simulations curve (the efficiency claim).

### Phase E: SHAP-Based Design Rule Extraction (First for CeSRR SWS)
1. Global SHAP values across full dataset — identify dominant geometry parameters for each frequency band.
2. Local SHAP analysis around selected target frequencies.
3. Convert findings into **explicit design rules** for CeSRR geometry synthesis (e.g., "to shift resonance from X GHz to Y GHz, increase r1 by Z% while holding r3/r4 ratio fixed").
4. Cross-validate extracted rules against known EM physics of CeSRR topology.

---

## 6) Experimental Protocol

### Data Split Strategy
- **Interpolation split**: random train/val/test (80/10/10).
- **Extrapolation split**: hold out full frequency sub-bands as hard test (model never saw those bands during training).

### Metrics
1. Frequency prediction MAE and RMSE.
2. **Geometry feasibility rate**: percentage of outputs satisfying r1 > r2 > r3 > r4 and all dimensional bounds.
3. Simulation agreement score: predicted geometry vs. full-wave verified response.
4. **Active-learning efficiency curve**: error vs. number of full-wave simulations (key plot for the paper).

### Ablation Studies (Essential for Peer Review)
1. Without physics-informed loss (ordering + EM penalties removed) — quantifies gain of H1.
2. Without active learning (random sampling instead) — quantifies gain of H2.
3. Deterministic inverse regression vs. generative inverse (cVAE) — quantifies gain of H3.
4. Without SHAP post-analysis — shows interpretability is a value-add, not just decoration.

---

## 7) Expected Contributions (Publishable Claims)

1. **First** physics-informed inverse design framework for all-metal CeSRR-based slow wave structures, with the CeSRR geometric ordering constraint (r1 > r2 > r3 > r4) embedded directly in model training.
2. **First** application of uncertainty-guided active learning to reduce CST/HFSS simulation budget specifically for this vacuum electron device class.
3. **First** SHAP-derived geometry-frequency design rule set for CeSRR/all-metal SWS — providing interpretable engineering insight previously absent from this domain.
4. A validated generative inverse design module (cVAE) that produces multiple feasible CeSRR geometry candidates per target frequency, outperforming prior one-to-one regression approaches in both validity rate and design flexibility.

---

## 8) Suggested Paper Structure

1. **Introduction** — HPM/TWT SWS design challenge; why ML has not reached this device class; what this paper contributes
2. **Related Work** — AI/ML inverse design (SWS-specific), PINN in EM, active learning for simulation-driven design, SHAP in EM; explicit gap statement for all-metal CeSRR SWS
3. **CeSRR-Based SWS: Geometry, Physics, and Dataset** — unit-cell topology, the r1 > r2 > r3 > r4 physical constraint, dataset description
4. **Proposed Framework** — physics-informed cVAE + uncertainty-guided active learning + SHAP pipeline
5. **Experiments and Ablations** — baseline vs. proposed; with/without physics constraints; AL efficiency curve; SHAP design rules
6. **Discussion** — physical validity of extracted rules; scalability to other SWS topologies; limitations
7. **Conclusion and Future Work**

---

## 9) Timeline (16 Weeks)

| Weeks | Task |
|---|---|
| 1–2 | Data cleaning, frequency normalization, constraint validation, unified schema |
| 3–4 | Baseline forward/inverse models (XGBoost, MLP, MDN) |
| 5–8 | Physics-informed surrogate + cVAE generative inverse model |
| 9–11 | Uncertainty-guided active learning loop; AL efficiency curve generation |
| 12–13 | SHAP analysis; design rule extraction and EM cross-validation |
| 14–15 | Writing, figures, ablation tables, comparison with Paper 1 & 2 |
| 16 | Final revision and submission package |

---

## 10) Immediate Next Actions

1. Finalize clean machine-readable dataset from the XLSX sheet (both full-structure and unit-cell tables).
2. Audit the dataset for r1 > r2 > r3 > r4 ordering violations — report the violation rate (this becomes a data quality result in the paper).
3. Define parameter bounds and hard physical constraints for both geometry layers.
4. Implement baseline inverse regression (MLP + MDN) and report first benchmark.
5. Begin physics-informed + cVAE inverse prototype with the ordering-penalty loss term.

---

*This plan is framed to produce a novel, peer-review-defensible paper. The novelty is not a new ML method — it is the first ML treatment of all-metal CeSRR SWS for HPM/TWT using a combined physics-informed, data-efficient, and interpretable pipeline that closes three simultaneously open sub-gaps in this domain.*
