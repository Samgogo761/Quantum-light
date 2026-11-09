# Briefing for Prof. Cohen: squeezed-light control of Neel-odd second-harmonic response in a PT antiferromagnet

Date: 2026-10-29
Purpose: a candid project briefing, not a manuscript. Every headline number below is from the Fourier-endpoint-corrected Q6 archive. The 24 October Chinese slides still quote the obsolete values 5.80 and 106.9 degrees. The frozen values are 4.98 and 110.6 degrees. Please do not send the old deck.

## One-sentence position

We have not produced a magnetic high-harmonic prediction for 1600 nm bright squeezed vacuum. What we have is a controlled eight-band PT antiferromagnet in which truncated QED shows that the squeeze angle of a low-occupation displaced squeezed state can steer the finite-time Neel-odd H2 current, and in which that steering can be split into input-statistic interference, Neel/SOC gating, finite-time electronic dynamics, and a residual quantum-mapping correction.

---

## 0. Where the project stands

This is a candidate theoretical-mechanism paper: a reproducible numerical phenomenon, a testable effective relation, and explicit boundaries. It is not a quantitative prediction for bilayer CrI3 driven by experimental BSV, and it is not yet a submission whose novelty we would claim with confidence.

| Layer | Status | Evidence |
|---|---|---|
| Phenomenon | obtained | Q6 QED: squeeze angle changes the Neel-odd H2 current |
| Effective formula | obtained | J_2^o(theta) ~ B + A e^{i theta} + C e^{-i theta} |
| Input control parameter | obtained | R = |M|/alpha^2 = 1.273 gives O(1) interference |
| Electronic channel | partial | Neel and SOC open the odd channel; inside the insulating window they do not pull arg(A/B) off pi |
| Absolute phase | partial | a same-protocol finite-time kernel accounts for 73.7% of the original 79 degree mismatch; Ehrenfest covers most of the rest |
| Real-material numbers | not done | 1600 nm only shows that a magnetic H2 channel exists at a few k points |
| Experimental source mapping | not done | the main result remains 24.8 um and 0.34 photons per mode |

I agree that the archive is now in a form that can be written up for an advisor. I do not agree that it is close to experimental BSV, and I will not repeat an unverified claim that a 1550 nm, about 5 dB displaced-squeezed experiment already covers this protocol.

---

## 1. The scientific question

The project is not: whether squeezed light beats a coherent drive in HHG yield, whether an antiferromagnet can produce SHG, or whether the output field is nonclassical.

The question we can actually defend is:

How do the coherent displacement alpha^2 and the anomalous correlator M(theta) of the incident light, once they pass through electronic channels controlled by Neel order and spin-orbit coupling, become a predictable Neel-odd second-harmonic complex amplitude and phase?

Causal chain we are trying to close:

input quantum statistics (alpha, M, theta) -> Neel/SOC-gated electronic response kernel -> Neel-odd H2 current

The magnet supplies a domain-odd readout. The light supplies a rotatable second-order correlator, not only a mean field.

---

## 2. Why an eight-band model first

The long-term material target remains bilayer antiferromagnetic CrI3. Sun et al., Nature 572, 497 (2019) already showed that the lattice is centrosymmetric and that the giant nonreciprocal SHG comes from layered AFM order, which jointly breaks inversion and time reversal. "AFM can produce SHG" is therefore not a result of ours.

To ask how quantum statistics enter that magnetic channel, we needed a Hamiltonian small enough for joint electron-photon evolution and still large enough to keep Neel order, interlayer coupling, and SOC. The eight-band M0 model is two layers x two sublattices x two spins. In-plane nearest-neighbor hopping is t_h = 1 eV. Interlayer same-site/opposite-site couplings are u, v. A Kane-Mele-type next-nearest-neighbor SOC is present. Opposite-sign exchange fields sit on the two layers. Momentum is restricted to one pair, k = (0.173, 0.271) and -k, sharing a single photonic mode.

M0 is a phenomenological PT-AFM Hamiltonian. It is not an eight-band downfolding of the CrI3 full112 Wannier model. Filling its gap, exchange, or SOC with CrI3 numbers does not make the calculation a quantum-HHG prediction for the real material.

The Wannier line (M52 / M88 / full112 / WINDOW144) is only a material check. At 1600 nm, coherent few-k runs show that a magnetic H2 channel exists. On M52 the amplitude drops by about 6.6 from 2x2 to 4x4. WINDOW144 still leaves a completeness question on full112. None of that licenses writing M0 as CrI3.

---

## 3. Why 24.8 um, alpha = 0.5, r = 0.3

This is the question a human referee, and Prof. Cohen, will ask first.

### 3.1 Three optical protocols must stay separate

| | Eight-band Q6 (main result) | 1600 nm M0 statistical scout | Published BSV-HHG experiment |
|---|---|---|---|
| Wavelength | 24.8 um | 1600 nm | 1600 nm |
| Photon energy | 0.05 eV | 0.775 eV | about 0.775 eV |
| Optical cycle | 82.71 fs | 5.337 fs | femtosecond |
| State | D(0.5) S(0.3 e^{i theta}) |0> | idealized high-gain DS / BSV / coherent | bright squeezed vacuum, coherent control |
| Mean occupation | 0.343 / mode | macroscopic | about 2e12 / pulse in the LN/a-Si experiment |
| Squeezing | r = 0.3 (ideal about 2.61 dB) | formal r ~ 11 | high-gain single-spatiotemporal-mode BSV |
| Time protocol | 1+4+1 cycle coupling switch, about 496 fs | 25 fs Gaussian | 25 fs BSV |
| Intensity | no W/cm^2 calibration | 1e10 W/cm^2 as a theory filter | experimental strong field, not a CrI3 safety point |
| Method | truncated QED + GH5 | Wigner-statistical SBE, quadrature not fully certified | experiment |
| Momentum | one k/-k pair | BZ scouts | real crystal |

Q6 uses g = 0.005 A^{-1} and omega_0 = 0.05 t_h. In the protocol name Qin_a2_141, a2 is the lattice polarization direction. It is not alpha ~ 1.41.

The experimental column follows Rasputnyi et al., Nat. Phys. 20, 1960 (2024). That intensity is one working point of that experiment. It must not be copied onto CrI3.

### 3.2 Why these numbers were chosen

Four constraints, none of them "this already looks like the experiment."

1. Hilbert-space size for truncated QED. The electronic sector is 36-dimensional, the photon cutoff is n_max = 8 (nine Fock states), and the joint space is 11664-dimensional. That is near the limit of a multi-cycle exact propagation we can still afford. A macroscopic Fock space is inaccessible.
2. The squeeze angle has to matter. |M| = (1/2) sinh(2r) = 0.318 and alpha^2 = 0.25, so R = |M|/alpha^2 = 1.273. When R = O(1), displacement and the anomalous correlator can compete. A later 1600 nm candidate that put 80% of the energy in displacement and 20% in squeezing sits at R ~ 0.25, where the squeeze angle is almost a perturbation. That is a different statistical window, not a failed magnification of Q6.
3. Deep detuning, on purpose. 2 omega_0 = 0.10 eV, while the local gap at the production k point is about 1.91 eV. Q6 studies a finite-time nonlinear complex response far from interband resonance. It does not study the 1600 nm window in which H2 sits at 1.55 eV and can meet real transitions.
4. A nonzero mean field is required if alpha^2 and M are to interfere in phase. Pure BSV has <a> = 0. Q6 is a displaced squeezed state: the mean displacement is held fixed and only the noise ellipse rotates. That is the clean comparison "same mean field and same mean photon number, only the fluctuation axis changes."

### 3.3 Has anyone prepared this light?

Displaced squeezed states are textbook Gaussian states. Ideal squeezing at r = 0.3 is modest. Continuous-variable and quantum-metrology experiments have prepared squeezed and displaced-squeezed light. That is not the same statement as "a femtosecond source of this kind has been focused onto a magnetic sample and used to drive SHG." I have not independently verified a 1550 nm, about 5 dB displaced-squeezed solid-state SHG experiment, and I will not cite one in this briefing. 24.8 um (about 12.1 THz) is also not the working wavelength of current solid BSV-HHG.

The honest line to a specialist: Q6 is a mechanism microscope. It is not a scaled-down copy of an experimental protocol.

---

## 4. Literature map

This section is written for someone who already knows the quantum-light HHG papers. It is a boundary map, not a tutorial.

### 4.1 Magnetic SHG

Sun et al., Nature 572, 497 (2019). Giant nonreciprocal SHG of bilayer AFM CrI3 comes from layered magnetic order. We borrow the fact that a Neel-odd signal can be read by SHG. That borrowing is not novelty.

First-principles work on bilayer CrI3 shows that some AFM stackings still have nonzero SHG with SOC switched off, and that SOC often enhances rather than uniquely creates the response. Therefore "SOC = 0 kills the odd channel in M0" belongs to this eight-band model. It is not a universal selection rule for all CrI3 magnetic SHG.

### 4.2 Quantum-light-driven solid HHG

Gorlach et al., Nat. Phys. 19, 1689 (2023). Quantum states of the driving field enter extreme nonlinear optics. We take from it that HHG can depend on photon statistics, not only on the mean field.

Rasputnyi et al., Nat. Phys. 20, 1960 (2024). 1600 nm, 25 fs bright squeezed vacuum drives nonperturbative HHG in LiNbO3 and amorphous silicon, compared with a coherent drive of the same mean intensity. The important difference for us: that experiment is high-gain pure BSV in nonmagnetic solids. It is not a low-occupation displaced squeezed state in a magnet.

Li et al., Phys. Rev. A 112, 033507 (2025). Quantum-optical features of solid HHG driven by squeezed coherent states. This is one of the closest theoretical precedents. "Squeezed coherent light changes solid harmonics" is no longer an empty niche.

Prabhu and Dixit, Phys. Rev. B 114, L140301 (2026). Forbidden harmonics in graphene as a sensor of squeezing fluctuations, including polarization and photon statistics. By 2026, using harmonics to read quantum fluctuations, and using squeeze phase to change polarization, is already in the literature.

### 4.3 Method background

de-la-Pena et al., J. Chem. Theory Comput. 21, 283 (2025). Multi-trajectory Ehrenfest versus exact QED. Our local feedback implementation is not identical.
Kane and Mele, Phys. Rev. Lett. 95, 226801 (2005). Only the structural form of next-nearest-neighbor SOC is borrowed.
Walls and Milburn, Quantum Optics (2008). Displacement and squeeze operators and second-moment identities.

### 4.4 What may still be ours

If anything remains, it is not "quantum light can change HHG." It is the mapping of Delta n = 0 and Delta n = +/- 2 coherence blocks of a displaced squeezed drive onto Neel-odd finite-time response coefficients A, B, C, together with a split of which pieces come from the input geometry R, which from the Neel/SOC channel, and which from finite-time electronic dynamics.

A literature search to date has not found an experiment or an equivalent theory that puts a controlled displaced squeezed state or BSV onto bilayer AFM CrI3 and studies this A/B/C decomposition. That is a search judgment, not a proof that no such work exists.

---

## 5. Model, protocol, and observable

The same incident light, the same polarization, and the same time reference are used for two static Hamiltonians that differ only by a global flip of the layer exchange fields, +N and -N. The sum and difference of the currents are the Neel-even and Neel-odd components. Odd/even here refers to the Neel vector, not to the harmonic order q, so H2 can still have a Neel-odd piece. N is a static label. The light and the two domains are not placed in a superposition.

A Peierls substitution shifts crystal momentum by g s(t) e X-hat, with X-hat the field operator. One photonic mode couples to both k and -k. The initial state is the electronic ground state of the uncoupled Hamiltonian tensored with the chosen optical state.

s(t) is a coupling switch: one-cycle rise, four-cycle plateau, one-cycle fall. It is not the electric-field envelope of a freely propagating experimental pulse. Q6 must not be described as a 496 fs laser of some known W/cm^2.

The main analysis window is [2 T_0, 4 T_0]. A positive-frequency Fourier projection of the odd current near q = 2 is called the H2 neighborhood. The result is complex. The two-dimensional amplitude is sqrt(|Jx|^2 + |Jy|^2). Neighboring windows still change by 16%-39%. This is a finite-time complex response, not a converged far-field harmonic yield, and not a photon count.

---

## 6. Methods

The same Hamiltonian, the same input, and the same window are treated in two ways.

1. Truncated QED: the joint electron-photon state is propagated, including back-action.
2. GH5: the input Wigner function is a correlated two-dimensional Gaussian, sampled on a 5x5 Gauss-Hermite grid. Each node is an independent mean-field trajectory with current feedback, then reweighted. GH5 is a quadrature order, not five trajectories.

GH5 is used for prediction and parameter scans. QED is the check. A GH5-QED mismatch measures approximation error. It does not by itself prove a quantum advantage. After the Fourier-endpoint repair, six analytic tests pass. The old-slide numbers 5.80 / 106.9 degrees came from the pre-repair main window and are retired.

---

## 7. Frozen headline results

Production input: |psi_theta> = D(0.5) S(0.3 e^{i theta}) |0>.

| Quantity | Value | How to read it |
|---|---:|---|
| 2D odd-current amplitude ratio theta=pi / theta=0 | 4.98 | finite-window complex amplitude, not a yield |
| Jx phase difference under covariance reversal | 110.6 deg | theta=pi/2 versus 3pi/2: same initial variance, opposite covariance |
| R = |M|/alpha^2 | 1.273 | displacement and anomalous correlator already compete |
| rho = |A|/|B| | 1.456 | QED, not the GH5 rho |
| arg(A/B) | 177.65 deg | nearly diametric interference, mostly input geometry |
| B | (-3.3006 + 0.9537 i) x 10^{-8} | Delta n = 0 block |
| A | (+4.7434 - 1.5839 i) x 10^{-8} | Delta n = +2 block |
| |C/A| | 1.54% | the counter-rotating term is small |
| Neighbor-window complex change | 16%-39% | not a stable harmonic |
| SOC scan span of arg((A+C)/B) | 4.37 deg | negative result inside the insulating window |
| K_M / K_B | 0.975-1.008 | on GH5 the same kernel reads alpha^2 and M |
| Finite-time kernel | 73.7% | fraction of the original 79 deg adiabatic-versus-QED gap |
| Ehrenfest kick on/off | -16.54 deg | covers most of the remainder, not all of it |
| Main-window QED/GH5 K_B | 0.868 | odd-channel suppression of about 13% |
| Main-window QED/GH5 K_M | 1.003 | no enhancement of the M readout |

Exchange strength eta strongly rescales |A| and rotates arg(B), but the near-pi value of arg(A/B) stays pinned to the input geometry. A held-out theta = pi/4 prediction misses by 0.667%, 0.425%, and 1.037% of |A|.

---

## 8. How far the A/B/C explanation goes

J_2^o(theta) ~ B + A e^{i theta} + C e^{-i theta}

Layer 1, exact photon-number blocks. For fixed electronic dynamics and a fixed probe window, J = Tr(rho_ph K). B reads Delta n = 0, A reads +2, C reads -2. B can contain ordinary occupation, the vacuum block, and higher diagonal processes. It is not identically K_B alpha^2.

Layer 2, an effective second-order readout, verified but not an identity: J ~ K_B alpha^2 + K_M M(theta). On GH5, K_M/K_B ~ 1, so rho ~ R. QED moves rho/R from 1.00 to 1.144 because the odd-channel K_B is suppressed, not because K_M is enhanced. About 15% of the remainder lives in the Delta n = 0 block.

Layer 3, the R window. Interference is strong when R = O(1). Cancellation is strongest near R ~ 1. When R << 1 the squeeze angle is almost a perturbation. For an ideal single-mode high-gain state, R ~ n_s / n_d. Keeping R ~ 1 therefore does not, as a matter of input statistics, require fewer than one photon per mode. That is mathematical feasibility. It is not a certified experimental source, and it does not prove that a 1600 nm material kernel equals the Q6 kernel.

Layer 4, electronic gating. Neel order and SOC together open the odd channel and rescale ||K||. Delta = 0 is a definitional zero. SOC = 0 is dynamical gating, but the gap also drops to 1.066 eV. Inside the insulating window, an SOC scan does not pull arg(A/B) off pi (span 4.37 deg). That is a negative result, not a magnetic phase fingerprint.

Layer 5, absolute phase. Equilibrium chi^(2) does not predict Q6 arg(B). A finite-time Peierls kernel with the same 1+4+1 switch and the same window accounts for 73.7% of the original 79 deg gap. Ehrenfest covers most of the rest. A GH5-QED residual of about 4 deg remains. Do not write a strict additive phase identity.

The unclosed layer: we cannot yet first-principles-predict the full complex values of A and C from individual electronic matrix elements. What can be predicted independently is the angular form, the R window, the on/off of the channel, and the main origin of arg(B).

---

## 9. Claims we will not make

Stating the boundary first is safer than being asked for it.

- 4.98 is not a harmonic yield and not an experimentally measurable enhancement factor.
- We have not proved a quantum advantage. A classical stochastic field with the same second moments may still reproduce the mean current.
- The non-negative P-function bound |M_cl| <= n_fl constrains the input, not the nonclassicality of output H2.
- arg(A/B) ~ pi is mostly input geometry, not a magnetic fingerprint.
- The material does not preferentially read M.
- 1600 nm has not reproduced the 110 deg rotation.
- M0 is not a CrI3 downfolding. full112 is not proved complete.
- SOC = 0 is not a universal selection rule for magnetic SHG.
- The mean current is not the output photon statistics. C_JJ, g^(2), and genuine BSV nonclassicality belong to another paper.

---

## 10. How far this is from experimental BSV

The distance has four segments. They should not be collapsed into "we still need a parameter scan."

1. Source statistics: from a low-occupation DS with R = O(1) to a high-gain, phase-locked displaced squeezed state or to pure BSV. Mathematically R can stay of order one at high brightness. Experimentally one still has to solve locking, loss, multimode structure, and calibration.
2. Field normalization: Q6 g has not been mapped onto an experimental mode volume and W/cm^2.
3. Electronic dynamics: deep detuning at 24.8 um is not H2 at 1.55 eV. Even at the same R, K_B, K_M, and higher-order processes can change.
4. Material model: an audited Wannier/SBE calculation is required. Editing M0 parameters is not that calculation.

If a paper is written now, the title should stay inside quantum-light-controlled magnetic second-harmonic response in a PT antiferromagnet. The methods should state the finite-time window, the single k/-k pair, the low-occupation displaced squeezed state, and the absence of a classical positive-P output control.

---

## 11. Questions we would actually like to ask

These are written so a specialist can answer them. They are not "can this go to Nature Physics."

1. Against Gorlach 2023, Rasputnyi 2024, squeezed-coherent solid HHG, and the 2026 graphene forbidden-harmonic work, is an A/B/C decomposition on a Neel-odd channel still a distinct theoretical question, or will it be read as one more model demonstration of known quantum-light HHG?
2. Is a main calculation at 24.8 um and <n> < 1 acceptable to a referee if R is written as a transferable statistical control, or does the paper need 1600 nm high-gain statistical electron dynamics before it stands?
3. If the observable remains a finite-time complex current rather than far-field harmonic photons, what is the largest scientific risk?
4. If the next stage should move toward experiment, would you keep a high-brightness displaced squeezed state at R ~ 1, or go to pure BSV and change the observable to fluctuations or two-time correlators?
5. For bilayer CrI3, magnetic SHG is already strong. If one tried to control it with quantum statistics, is the realistic readout phase, polarization, or domain contrast?

---

## 12. What we will and will not do next

Now: freeze the corrected numbers, finish this briefing and the English seminar slides, and keep the mechanism boundaries explicit. No further SOC scans, no denser M52 Brillouin-zone meshes, and no n_max = 10 production QED for the sake of a submission.

Only if this discussion concludes that the mechanism itself is worth a paper would we enlarge the computation:

1. Revise the English write-up until it can be discussed with an advisor.
2. If experiment is the next target: first audit 1600 nm single-mode field normalization and whether a high-gain Wigner quadrature at R ~ 1 is trustworthy. If the quadrature fails, no material scan.
3. Only if a magnetic phase or polarization candidate cannot be explained as a reweighting of input second moments would we touch an audited Wannier model.

Not now: full-BZ exact Fock QED, output g^(2), a classical positive-P output control, a new WINDOW144 Wannier construction, or putting 4.98 into an abstract as an enhancement factor.

---

## 13. Suggested oral order (about 20 minutes)

1. Question: if the mean field and the mean photon number stay fixed and only the squeeze axis rotates, does the Neel-odd H2 current change?
2. Model: eight-band PT-AFM, not a CrI3 downfolding. Light: 24.8 um low-occupation displaced squeezed state, not experimental BSV.
3. Result: QED gives a finite-window amplitude ratio 4.98 and a covariance-reversal phase of 110.6 degrees.
4. Formula: B + A e^{i theta} + C e^{-i theta}. R = 1.273 is why the interference is strong.
5. What the magnet does: it opens the Neel-odd channel and sets arg(B). arg(A/B) ~ pi is mostly input geometry.
6. Boundaries: neighbor windows 16%-39%. No quantum advantage. No 1600 nm reproduction.
7. Ask: is this mechanism paper worth writing, and should the next calculation be experimental-scale R ~ 1 or a finished eight-band theory manuscript?

---

## 14. Files behind this briefing

- Chinese counterpart: docs/04_BSV量子光/cohen_briefing_20261029/Cohen讨论_课题详解_中文_20261029.md
- Frozen conclusions: docs/04_BSV量子光/线Q_外审后结论_20261029.md
- Paper skeleton: docs/04_BSV量子光/线Q_论文骨架_20261029.md
- Theory draft: docs/04_BSV量子光/线Q_理论节草稿_20261029.md
- Figures: results/q_line/paper_figures_20261029/figures/fig1 through fig8
- Retired internal slides (obsolete numbers): 汇报_量子输入磁序复响应_Q6版_20261024.pptx

The accompanying English seminar deck uses the same frozen numbers and the same eight figures. It is written as a discussion with a specialist in quantum-light HHG, not as a claim that the eight-band far-infrared protocol is an experimental BSV source.
