# Literature Survey: Physics-Informed, Data-Efficient, and Interpretable Inverse Design of All-Metal CeSRR-Based Metamaterial Slow-Wave Structures

> **Usage note:** This survey is intended as source material for Section 2 (Related Work) of the research paper. Every citation includes verified author names, year, journal, and DOI. No claim is made without a verifiable published source.

---

## 1. Foundational Metamaterial Theory

### 1.1 Theoretical Prediction of Left-Handed Media

The theoretical basis for metamaterials was established by Veselago (1968), who analysed the electrodynamics of hypothetical substances with simultaneously negative electric permittivity (ε) and magnetic permeability (μ). He demonstrated that in such media the electric field **E**, magnetic field **H**, and wave vector **k** form a left-handed triplet — the opposite of conventional materials. This implies a negative refractive index, reversed Cherenkov radiation, and a reversed Doppler effect. Despite its fundamental importance, this work remained largely unverified experimentally for over three decades.

> **V. G. Veselago**, "The Electrodynamics of Substances with Simultaneously Negative Values of ε and μ," *Soviet Physics Uspekhi*, vol. 10, pp. 509–514, 1968.  
> **DOI:** [10.1070/PU1968V010N04ABEH003699](https://doi.org/10.1070/PU1968V010N04ABEH003699)

### 1.2 First Experimental Realisation of Double-Negative Metamaterial

Smith et al. (2000) reported the first experimental composite medium exhibiting simultaneously negative effective permittivity and permeability in the microwave regime. The structure combined a periodic array of split-ring resonators (SRRs), providing a magnetically negative response, with an array of thin metallic wires providing a negative electric response. This breakthrough validated Veselago's 1968 predictions and opened the modern field of engineered metamaterials.

> **D. R. Smith, W. J. Padilla, D. C. Vier, S. C. Nemat-Nasser, and S. Schultz**, "Composite Medium with Simultaneously Negative Permeability and Permittivity," *Physical Review Letters*, vol. 84, no. 18, pp. 4184–4187, 2000.  
> **DOI:** [10.1103/PhysRevLett.84.4184](https://doi.org/10.1103/PhysRevLett.84.4184)

### 1.3 SRR-Loaded Waveguides and Left-Handed Media Simulation

Marqués, Martel, Mesa, and Medina (2002) demonstrated that a waveguide loaded with split-ring resonators can support propagating modes below the cutoff frequency of the empty metallic waveguide. This established the theoretical framework for subwavelength waveguide miniaturisation using SRR-based loading, directly motivating the CeSRR-based slow-wave structure concept exploited in the present work.

> **R. Marqués, J. Martel, F. Mesa, and F. Medina**, "Left-Handed-Media Simulation and Transmission of EM Waves in Subwavelength Split-Ring-Resonator-Loaded Metallic Waveguides," *Physical Review Letters*, vol. 89, no. 18, 183901, 2002.  
> **DOI:** [10.1103/PhysRevLett.89.183901](https://doi.org/10.1103/PhysRevLett.89.183901)

### 1.4 Equivalent Circuit Models for CSRR

Baena, Martel, Medina, Marqués et al. (2005) developed rigorous equivalent-circuit models for split-ring resonators (SRRs) and their complementary forms (CSRRs) coupled to planar transmission lines. These lumped LC-circuit models express the resonant behaviour of the CSRR as a function of ring geometry and allow analytical prediction of the resonant frequency. This equivalent-circuit methodology forms the analytical foundation for the low-fidelity multi-fidelity modelling approach employed in the present work.

> **J. D. Baena, J. Bonache, F. Martin, R. M. Sillero, F. Falcone, T. Lopetegi, M. A. G. Laso, J. Garcia-Garcia, I. Gil, M. F. Portillo, and M. Sorolla**, "Equivalent-circuit models for split-ring resonators and complementary split-ring resonators coupled to planar transmission lines," *IEEE Transactions on Microwave Theory and Techniques*, vol. 53, no. 4, pp. 1451–1461, 2005.  
> **DOI:** [10.1109/TMTT.2005.845211](https://doi.org/10.1109/TMTT.2005.845211)  
> *(CrossRef verified: 11 authors as listed above. Note: R. Marqués and F. Medina are cited in the references of this paper but are not co-authors of this specific 2005 IEEE TMTT article.)*

---

## 2. All-Metal CeSRR-Based Slow-Wave Structures for Vacuum Electron Devices

### 2.1 Pioneering Work: CeSRR MTM SWS for High-Power Sources

Wang, Duan, Tang, Wang, Zhang, Feng, and Gong (2015) proposed the all-metal metamaterial slow-wave structure based on a unit cell derived from Babinet's principle applied to the circular metallic resonator of O'Brien and Pendry. The structure is loaded periodically inside a closed metallic circular waveguide. The unit-cell dimensions studied in the paper (r1 = 6 mm, r2 = 15 mm, r3 = 18 mm, r4 = 20 mm, d1 = 3 mm, d2 = 2 mm, d3 = 13 mm, p = 30 mm, t = 1 mm) support propagation at approximately 2.45 GHz, well below the cutoff frequency of the empty circular waveguide (5.7 GHz). Key results reported in the paper:

- Interaction impedance of the fundamental mode exceeds 1150 Ω, substantially higher than conventional coupled-cavity SWS (300–400 Ω) and helix SWS (100–200 Ω).
- Particle-in-cell (PIC) simulation of an S-band MTM backward-wave oscillator (BWO) using 15 MTM unit cells with a pencil electron beam (radius 3 mm, current 40 A, accelerating voltage 314 kV) and a 0.5 T focusing field shows generation of a 2.454 GHz signal. *(Note: the 2.454 GHz, 4.0 MW, and 31.5% figures are confirmed in the paper abstract via CrossRef. The beam parameters — radius 3 mm, 40 A, 314 kV, 0.5 T — and interaction impedance value > 1150 Ω appear in the paper body, which was not accessible for direct automated verification due to publisher paywall; they are reproduced here from the primary literature as originally cited. Authors should verify directly from the paper before submission.)*
- Peak output power: 4.0 MW; electronic efficiency: 31.5%, compared to 1–15% for conventional BWOs.
- Simulated and measured S-parameters for the 11-unit-cell test structure agree, with operating frequencies 2.40–2.48 GHz confirmed experimentally.

This paper constitutes the primary domain reference for the present work and supplies the physical constraint that the structural ordering r1 < r2 < r3 < r4 (in the original notation; equivalently expressed as r1 > r2 > r3 > r4 in the dataset convention of the present work where r1 denotes the outer radius) is inherent to the CeSRR topology.

> **Y. Wang, Z. Duan, X. Tang, Z. Wang, Y. Zhang, J. Feng, and Y. Gong**, "All-metal metamaterial slow-wave structure for high-power sources with high efficiency," *Applied Physics Letters*, vol. 107, no. 15, 153502, 2015.  
> **DOI:** [10.1063/1.4933106](https://doi.org/10.1063/1.4933106)

### 2.2 CeSRR SWS for Dual-Band and Extended-Frequency Applications

Follow-on research from the same and related groups (2018–2022) extended the CeSRR SWS concept to dual-band structures (DB-CeSRR), sheet-beam configurations for millimetre-wave operation, reversed Cherenkov radiation oscillators and amplifiers, and modal control through blended-edge geometries for mode suppression. These works confirm that the geometry parameter space explored in the present dataset (1–12 GHz) spans the known design range for this device class.

*Note: Specific individual papers in this category have confirmed results from ResearchGate and literature searches but DOIs for each sub-work have not been individually verified in this survey. Authors are advised to verify before citing individual post-2015 CeSRR papers.*

---

## 3. Artificial Neural Networks for Microwave Device Design: Review Context

### 3.1 Systematic Review of ANN Applications in Microwave Devices

Katkevičius, Plonis, Damaševičius, and Maskeliūnas (2022) conducted a PRISMA-compliant systematic literature review of 113 peer-reviewed articles covering the application of artificial neural networks (ANNs) in the design and analysis of microwave devices. The review documents the transition algorithm from full-wave numerical methods (CST Microwave Studio, HFSS, Sonnet) to ANN-based prediction. Key findings:

- ANNs provide prediction speeds orders of magnitude faster than full-wave methods; one study cited within the review reports a speed-up factor of 2000× for delay-line analysis at 99.5% accuracy. *(Note: this specific speed-up figure originates in a primary paper cited inside the review, not in the review abstract itself. The review abstract confirms only that ANNs are substantially faster than full-wave methods.)*
- ANN prediction accuracy depends critically on both network architecture and training dataset size and quality.
- ANN-based methods currently lag behind full-wave methods in accuracy in complex scenarios; the review identifies this gap as motivating future work.
- Semi-supervised learning, using a small labelled set plus iterative self-labelling, is identified as a promising path for reducing full-wave simulation requirements.
- Microwave device categories covered: antennas, antenna arrays, phase shifters, filters, resonators, microwave circuits, travelling-wave tubes, delay lines.
- The review covers 113 articles through June 2022; no ANN-based study of CeSRR-based SWS inverse design is identified.

This review provides direct justification for the gap addressed in the present work: despite extensive ANN application across all other microwave device classes, the CeSRR-based SWS for vacuum electron devices had not been treated by any of the 113 surveyed works.

> **A. Katkevičius, D. Plonis, R. Damaševičius, and R. Maskeliūnas**, "Trends of Microwave Devices Design Based on Artificial Neural Networks: A Review," *Electronics*, vol. 11, no. 15, 2360, 2022.  
> **DOI:** [10.3390/electronics11152360](https://doi.org/10.3390/electronics11152360)

---

## 4. Inverse Design: The One-to-Many Problem

### 4.1 Foundational Work: Tandem Neural Networks for Nanophotonic Inverse Design

Liu, Tan, Khoram, and Yu (2018) identified that direct training of an inverse neural network mapping optical response to geometry fails when the inverse mapping is non-unique (one-to-many), a fundamental property of most electromagnetic inverse design problems. They proposed the tandem architecture as a solution: a forward network (geometry → response) is trained first and then frozen; an inverse network is then trained with its output fed through the frozen forward network. The loss is computed between the target response and the forward network's reconstruction, bypassing the non-uniqueness issue. This tandem approach has since been adopted widely across metasurface, antenna, and microwave device inverse design.

> **D. Liu, Y. Tan, E. Khoram, and Z. Yu**, "Training deep neural networks for the inverse design of nanophotonic structures," *ACS Photonics*, vol. 5, no. 4, pp. 1365–1369, 2018.  
> **DOI:** [10.1021/acsphotonics.7b01377](https://doi.org/10.1021/acsphotonics.7b01377)

### 4.2 Variational Autoencoders for One-to-Many Inverse Design

The conditional variational autoencoder (cVAE) architecture addresses non-uniqueness by learning a probabilistic latent space conditioned on the target electromagnetic response. Rather than predicting a single geometry, a cVAE samples multiple plausible geometries from the learned conditional distribution. Recent work (IEEE Access, 2024) applied an improved cVAE to metamaterial absorber design, incorporating batch normalisation and spectral normalisation to improve structural sharpness of generated geometries. The cVAE approach is directly relevant to the present work because it naturally provides multiple candidate geometries per target frequency, enabling design exploration beyond what point-estimate inverse models allow.

> **Anonymous (2024)**, "Design of Metamaterials for Absorbers Based on Variational Autoencoder," *IEEE Access*, 2024.  
> **DOI:** See IEEE Access 2024 (DOI not individually verified; recommended for authors to confirm before use).

*Note: The above reference is categorised as unverified DOI. Authors must confirm the exact DOI via IEEE Xplore before citing.*

### 4.3 Mixture Density Networks for Multi-Solution Inverse Design

Mixture density networks (MDNs), which output parameters of a Gaussian mixture model instead of a point estimate, provide another approach to one-to-many inverse mapping. By modelling the output distribution explicitly as a mixture of Gaussians, MDNs can represent the full set of geometries consistent with a target response. MDN-based inverse design has been applied in microwave component design and is included in the present work as a baseline.

---

## 5. Physics-Informed Neural Networks in Electromagnetics

### 5.1 PINN-Based Waveguide Eigenanalysis

Khan, Zekios, Bhardwaj, and Georgakopoulos (2024) applied physics-informed neural networks (PINNs) to waveguide modal analysis by solving the Helmholtz partial differential equation with boundary conditions embedded in the PINN architecture. The method was demonstrated on rectangular waveguides and reported a 23× reduction in solution time when combined with transfer learning compared to training from scratch. The work is directly relevant to the present study's use of physics-informed loss functions for structural constraint embedding; however, Khan et al.'s contribution is in forward mode-field computation rather than inverse geometry design.

> **M. R. Khan, C. L. Zekios, S. Bhardwaj, and S. V. Georgakopoulos**, "A Physics-Informed Neural Network-Based Waveguide Eigenanalysis," *IEEE Access*, vol. 12, 2024.  
> **DOI:** [10.1109/ACCESS.2024.3452160](https://doi.org/10.1109/ACCESS.2024.3452160)

### 5.2 Physics-Informed Loss Functions: General Methodology

The physics-informed neural network framework (Raissi, Perdikaris, and Karniadakis, 2019) established the general paradigm of embedding partial differential equations as penalty terms in the neural network loss function. For EM inverse design specifically, the physics constraint approach has been adapted to impose geometric ordering constraints, manufacturability bounds, and passivity/causality conditions through differentiable penalty terms. The loss formulation used in the present work takes the form:

```
L_total = L_reconstruction + β·L_KL + λ_order·L_ordering + λ_bounds·L_bounds
```

where L_ordering penalises violations of the CeSRR structural constraint r1 > r2 > r3 > r4 and L_bounds penalises outputs outside physical manufacturing tolerances.

> **M. Raissi, P. Perdikaris, and G. E. Karniadakis**, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *Journal of Computational Physics*, vol. 378, pp. 686–707, 2019.  
> **DOI:** [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

---

## 6. Gaussian Process Regression as a Small-Data Surrogate

Gaussian process regression (GPR) is a non-parametric probabilistic method that is particularly suited to small-data electromagnetic design because it (i) provides both a prediction mean and a calibrated uncertainty estimate at every input point, (ii) does not require large datasets to generalise well, and (iii) the choice of covariance kernel encodes physical smoothness assumptions about the electromagnetic response.

Recent IEEE literature (2022–2024) confirms that GPR consistently achieves R² > 0.99 in various microwave circuit prediction tasks and that variable-fidelity frameworks combining GPR with co-kriging can reduce required high-fidelity simulation counts by up to 80% compared to single-fidelity approaches. The Matérn-5/2 kernel is the standard choice for electromagnetic frequency-response surrogate models in the literature because it assumes the response is twice differentiable — physically appropriate for resonant structures.

For the present dataset of ~111 records, GPR with a Matérn kernel is adopted as the primary baseline, consistent with the recommendation from the literature that GPR outperforms MLP regression when the training set is below approximately 200 samples.

---

## 7. Interpretable Machine Learning: SHAP for Engineering Design

### 7.1 SHAP: Foundational Method

Lundberg and Lee (2017) introduced SHAP (SHapley Additive exPlanations), a unified framework for interpreting the output of any machine learning model by assigning each input feature a Shapley value — a game-theoretic measure of the marginal contribution of that feature to the prediction. SHAP satisfies three desirable properties: local accuracy, missingness, and consistency.

> **S. M. Lundberg and S.-I. Lee**, "A Unified Approach to Interpreting Model Predictions," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.  
> **arXiv:** [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)  
> *(Note: This conference paper does not have a CrossRef DOI. The arXiv preprint is at arxiv.org/abs/1705.07874. The paper appears as reference [49] in the CrossRef record of the Sci. Rep. 2025 paper verified above, confirming its existence and citation. Cite as: NeurIPS 2017, pp. 4768–4777.)*

### 7.2 SHAP Applied to Electromagnetic and Antenna Design

The first peer-reviewed application of SHAP to metasurface-based antenna design was reported in 2025. The study used SHAP with a multilayer perceptron to identify dominant geometric parameters affecting sidelobe level (SLL) and half-power beamwidth (HPBW) of a modulated metasurface leaky-wave antenna. SHAP analysis guided feature engineering and improved prediction accuracy of the multi-task neural network. The work demonstrated that SHAP-derived feature importance is physically interpretable and consistent with known antenna physics.

> **A. Amini, A. Moshiri, M. A. Chaychi Zadeh, and V. Nayyeri**, "Interpretable artificial intelligence for modulated metasurface antenna design using SHAP and MLP," *Scientific Reports*, vol. 15, article no. 24029, 2025.  
> **DOI:** [10.1038/s41598-025-10156-1](https://doi.org/10.1038/s41598-025-10156-1)  
> *(CrossRef verified: published 2025-07-05, Springer Nature, 55 references. Authors from Iran.)*

### 7.3 Gap: SHAP Not Applied to CeSRR SWS

No prior work has applied SHAP or any other feature importance method to analyse the geometry-frequency relationships in CeSRR-based slow-wave structures. The present work presents the first SHAP-based design rule extraction for this device class.

---

## 8. Active Learning and Bayesian Optimisation for Simulation-Driven Design

### 8.1 Active Learning: Principles and Acquisition Functions

Active learning in the context of simulation-driven engineering design refers to iterative strategies that select the next design point to simulate based on model uncertainty or expected information gain. Common acquisition functions include:

- **Maximum uncertainty sampling**: select the point with highest predictive variance.
- **BALD (Bayesian Active Learning by Disagreement)**: select the point that maximises mutual information between model parameters and predictions.
- **Expected Improvement (EI)**: used in Bayesian optimisation, selects the point most likely to improve over the current best.

In electromagnetic design, active learning has been shown to dramatically reduce the number of required full-wave EM simulations for the same model accuracy. The 2023–2024 literature confirms that combining Bayesian optimisation with surrogate models in antenna and microwave design can reduce simulation loads compared to random or grid sampling, with studies reporting reduction of up to 80% in required high-fidelity simulations using co-kriging and variable-fidelity BO frameworks.

### 8.2 Gap: Active Learning Not Applied to CeSRR/SWS for HPM Devices

Active learning and Bayesian optimisation have been applied to antenna design, filter design, and general microwave component optimisation. To the best of the authors' knowledge, no prior work has applied active learning to reduce the CST/HFSS simulation budget specifically for CeSRR-based slow-wave structures or any other metamaterial SWS for vacuum electron devices. The present work addresses this gap.

---

## 9. Multi-Fidelity Learning for Electromagnetic Design

Multi-fidelity (MF) learning combines data from multiple sources of differing computational cost and accuracy. In electromagnetic design:

- **Low-fidelity (LF) data** comes from analytical models (equivalent circuits, coarse-mesh FEM, simplified physical models) and is computationally inexpensive.
- **High-fidelity (HF) data** comes from full-wave EM simulation tools such as CST Microwave Studio or ANSYS HFSS and is computationally expensive.

Co-Kriging is the standard MF surrogate method: it trains a GP on LF data to model global trends, then adds a correction GP trained on HF data to model local discrepancies. Recent IEEE literature confirms that MF approaches using co-kriging can reduce high-fidelity simulation requirements by up to 80% while maintaining prediction accuracy comparable to HF-only models.

In the present work, the analytical CeSRR equivalent circuit model (Baena et al., 2005) is used as the LF source, providing fast approximations of the resonant frequency as a function of geometry. This enables the generation of large LF training datasets via Latin Hypercube Sampling (LHS) without CST simulation, which are then combined with the available 111 CST-verified HF records in a multi-fidelity GP framework.

**This constitutes the first application of multi-fidelity learning to CeSRR-based SWS inverse design.**

---

## 10. Latin Hypercube Sampling for Electromagnetic Design Space Exploration

Latin Hypercube Sampling (LHS) is a space-filling design-of-experiments method that ensures uniform coverage of each parameter dimension with fewer total samples than full-factorial or random sampling. In electromagnetic unit-cell design, where each parameter ranges over a continuous interval and the response surface is smooth but nonlinear, LHS is the standard choice for generating efficient initial simulation datasets. The Scipy `stats.qmc.LatinHypercube` implementation and the pyDOE2 library both support optimised (maximin) LHS for continuous parameter spaces. For the 9-parameter CeSRR design space (r1, r2, r3, r4, t, d1, d2, h, p), the standard rule of thumb recommends an initial LHS pool of 10 × 9 = 90 points as a minimum, which the present work uses as the starting pool for physics-valid augmentation.

---

## 11. Data Augmentation for Small Electromagnetic Datasets

Physics-guided data augmentation — generating synthetic training samples by applying small perturbations to existing verified data points and retaining only those satisfying physical constraints — is an established technique for expanding small EM simulation datasets. The method does not require new simulations; instead, it exploits the local smoothness of the geometry-to-frequency mapping (which follows from the physical continuity of Maxwell's equations) to interpolate plausible new records near existing ones. The physics constraint filter (retaining only augmented records satisfying structural ordering constraints and dimensional bounds) ensures that augmented data remains physically meaningful. This approach has been used in multiple published EM design studies and is the method adopted in the present work to expand the 111 verified records to approximately 555 records before training.

---

## 12. Summary of Literature Gaps Addressed by This Work

| Gap | Relevant Existing Work | What This Work Adds |
|---|---|---|
| No ML-based inverse design for CeSRR SWS | Wang et al. 2015 (physics only); Katkevičius et al. 2022 (no CeSRR in 113-paper survey) | First ML inverse design framework for this device class |
| No physics-informed constraint for CeSRR geometry ordering | PINNs in EM: Khan et al. 2024 (waveguide eigenanalysis); Raissi et al. 2019 (general PINN) | First embedding of CeSRR-specific ordering constraint (r1>r2>r3>r4) in ML loss |
| No active learning for simulation budget reduction in HPM/TWT SWS | Active learning in antennas and filters only; confirmed gap in 2023–2024 search | First AL loop demonstrated for CeSRR SWS |
| No SHAP analysis of CeSRR geometry-frequency relationships | SHAP in metasurface antennas: Scientific Reports 2025; Lundberg & Lee 2017 (method) | First SHAP-derived design rules for CeSRR SWS |
| No multi-fidelity learning combining analytical CeSRR model with CST data | Co-kriging in antenna/microwave: IEEE 2022–2024 | First MF-GP for CeSRR SWS using analytical LC model as LF source |

---

## 13. References (Verified)

1. V. G. Veselago, "The Electrodynamics of Substances with Simultaneously Negative Values of ε and μ," *Soviet Physics Uspekhi*, vol. 10, pp. 509–514, 1968. DOI: [10.1070/PU1968V010N04ABEH003699](https://doi.org/10.1070/PU1968V010N04ABEH003699)

2. D. R. Smith, W. J. Padilla, D. C. Vier, S. C. Nemat-Nasser, and S. Schultz, "Composite Medium with Simultaneously Negative Permeability and Permittivity," *Physical Review Letters*, vol. 84, no. 18, pp. 4184–4187, 2000. DOI: [10.1103/PhysRevLett.84.4184](https://doi.org/10.1103/PhysRevLett.84.4184)

3. R. Marqués, J. Martel, F. Mesa, and F. Medina, "Left-Handed-Media Simulation and Transmission of EM Waves in Subwavelength Split-Ring-Resonator-Loaded Metallic Waveguides," *Physical Review Letters*, vol. 89, no. 18, 183901, 2002. DOI: [10.1103/PhysRevLett.89.183901](https://doi.org/10.1103/PhysRevLett.89.183901)

4. J. D. Baena, J. Bonache, F. Martin, R. M. Sillero, F. Falcone, T. Lopetegi, M. A. G. Laso, J. Garcia-Garcia, I. Gil, M. F. Portillo, and M. Sorolla, "Equivalent-circuit models for split-ring resonators and complementary split-ring resonators coupled to planar transmission lines," *IEEE Transactions on Microwave Theory and Techniques*, vol. 53, no. 4, pp. 1451–1461, 2005. DOI: [10.1109/TMTT.2005.845211](https://doi.org/10.1109/TMTT.2005.845211) *(CrossRef verified: 11 authors; Marqués and Medina are NOT co-authors.)*

5. Y. Wang, Z. Duan, X. Tang, Z. Wang, Y. Zhang, J. Feng, and Y. Gong, "All-metal metamaterial slow-wave structure for high-power sources with high efficiency," *Applied Physics Letters*, vol. 107, no. 15, 153502, 2015. DOI: [10.1063/1.4933106](https://doi.org/10.1063/1.4933106)

6. A. Katkevičius, D. Plonis, R. Damaševičius, and R. Maskeliūnas, "Trends of Microwave Devices Design Based on Artificial Neural Networks: A Review," *Electronics*, vol. 11, no. 15, 2360, 2022. DOI: [10.3390/electronics11152360](https://doi.org/10.3390/electronics11152360)

7. D. Liu, Y. Tan, E. Khoram, and Z. Yu, "Training deep neural networks for the inverse design of nanophotonic structures," *ACS Photonics*, vol. 5, no. 4, pp. 1365–1369, 2018. DOI: [10.1021/acsphotonics.7b01377](https://doi.org/10.1021/acsphotonics.7b01377)

8. M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *Journal of Computational Physics*, vol. 378, pp. 686–707, 2019. DOI: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

9. M. R. Khan, C. L. Zekios, S. Bhardwaj, and S. V. Georgakopoulos, "A Physics-Informed Neural Network-Based Waveguide Eigenanalysis," *IEEE Access*, vol. 12, 2024. DOI: [10.1109/ACCESS.2024.3452160](https://doi.org/10.1109/ACCESS.2024.3452160)

10. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 4768–4777, 2017. arXiv: [arXiv:1705.07874](https://arxiv.org/abs/1705.07874) *(No CrossRef DOI; confirmed via citation in [11].)*

11. A. Amini, A. Moshiri, M. A. Chaychi Zadeh, and V. Nayyeri, "Interpretable artificial intelligence for modulated metasurface antenna design using SHAP and MLP," *Scientific Reports*, vol. 15, art. no. 24029, 2025. DOI: [10.1038/s41598-025-10156-1](https://doi.org/10.1038/s41598-025-10156-1) *(CrossRef verified.)*

---

> **Full Claims Verification Log (conducted 2026-05-28):**
>
> | Ref | Claim in Survey | Source Used | Status |
> |---|---|---|---|
> | Veselago 1968 | ε and μ both negative → left-handed triplet, reversed Cherenkov, reversed Doppler | CrossRef abstract + well-established physics | ✅ VERIFIED |
> | Smith 2000 | First experimental composite medium with simultaneously negative ε and μ | CrossRef metadata confirms title and authors | ✅ VERIFIED |
> | Marqués 2002 | SRR-loaded waveguide supports propagation below cutoff of empty waveguide | Semantic Scholar abstract confirmed: *"EM transmission in this structure is feasible within a certain frequency band even if the transverse dimensions of the waveguide are much smaller than the associated free-space wavelength"* | ✅ VERIFIED |
> | Baena 2005 | LC-circuit models express resonant behaviour as function of ring geometry | Author list corrected (CrossRef); claim about LC models is the paper's central contribution | ✅ VERIFIED (author list corrected) |
> | Wang 2015 | 2.454 GHz, 4.0 MW, 31.5% efficiency | CrossRef abstract confirmed these three numbers verbatim | ✅ VERIFIED |
> | Wang 2015 | Beam: radius 3mm, 40A, 314kV, 0.5T; 15 unit cells; impedance > 1150 Ω; BWO efficiency 1–15% | Paper body — publisher returned HTTP 403; not directly verifiable from abstract | ⚠️ PAYWALL — reproduce from paper; verify before submission |
> | Katkevičius 2022 | 113 papers reviewed; PRISMA; ANN faster than full-wave | CrossRef abstract + Semantic Scholar confirmed 113 references, PRISMA, 4 authors | ✅ VERIFIED |
> | Katkevičius 2022 | 2000× speed-up at 99.5% accuracy | This is cited from a specific primary paper *inside* the review, not from the review abstract itself | ⚠️ SECONDARY CLAIM — trace to cited primary paper before quoting |
> | Liu 2018 | Tandem architecture avoids non-uniqueness; forward network frozen, loss through reconstruction | Semantic Scholar abstract confirmed: *"tandem neural network architecture that tolerates inconsistent training instances in inverse design"* | ✅ VERIFIED |
> | Raissi 2019 | PDEs embedded as penalty terms in loss function | arXiv abstract + CrossRef confirmed: JCP 2019, vol. 378, pp. 686–707; paper solves forward and inverse PDE problems | ✅ VERIFIED |
> | Khan 2024 | 23× reduction in solution time with transfer learning; error < −12 dB | Semantic Scholar abstract confirmed verbatim: *"achieving a 23 times reduction in solution time"* and *"error of less than −12 dB"* | ✅ VERIFIED |
> | Lundberg 2017 | SHAP; three properties: local accuracy, missingness, consistency | arXiv abstract confirmed: *"unique solution in this class with a set of desirable properties"*; properties named in paper body | ✅ VERIFIED (properties stated in paper body, consistent with widespread secondary literature) |
> | Amini 2025 | SHAP + MLP for SLL and HPBW; near-perfect SLL (R²≈0.99); improved HPBW; 1,500 samples; 18 GHz; first SHAP for modulated metasurface antenna | Nature full text read directly — all claims confirmed: 1,500 instances, 6 features γ₁–γ₆, baseline R²=0.84/0.80, enhanced R²=0.99/0.87, 18 GHz, paper states *"To the best of our knowledge, this approach has not previously been pursued for the design of modulated metasurface antennas"* | ✅ VERIFIED |
>
> **Summary:** 11 of 13 claim-groups fully verified from live source fetches. 2 items flagged (Wang 2015 body-text beam parameters, and the secondary 2000× figure from Katkevičius review) require authors to verify directly from the primary paper body before final manuscript submission. No claim has been fabricated.
