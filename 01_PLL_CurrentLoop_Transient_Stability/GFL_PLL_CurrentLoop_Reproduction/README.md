# GFL current-loop–PLL transient synchronization reproduction

This package reconstructs the main mathematical results of:

> C. Wu, Y. Lyu, Y. Wang, and F. Blaabjerg, “Transient Synchronization Stability Analysis of Grid-Following Converter Considering the Coupling Effect of Current Loop and Phase Locked Loop,” *IEEE Transactions on Energy Conversion*, 39(1), 2024.

## What is reproduced

1. The conventional **second-order PLL synchronization model**, Eqs. (6)–(8).
2. The **fourth-order nonlinear model**, Eq. (18), with states
   \(x=[\delta,\Delta\omega,\Delta I_d,\Delta \dot I_d]\).
3. The paper’s key qualitative reversal at \(f_{BW,PLL}=50\) Hz:
   - current neglected: stable;
   - \(f_c=300\) Hz: loss of synchronism;
   - \(f_c=500\) Hz: resynchronization.
4. An EAC-like plot showing how dynamic current changes the equivalent input.
5. A reconstruction of Fig. 9 using a digitized lower boundary and the published conservative line
   \(f_c=9.461f_{BW,PLL}-74.2\).

## Reproducibility gap in the paper

An exact bit-for-bit reconstruction is not identifiable from the article alone:

- Table I leaves the current PI gains \(k_{pc}\) and \(k_{ic}\) as “–”.
- The mapping from the named bandwidth \(f_c\) to \((k_{pc},k_{ic})\) is not stated.
- The mapping from \(f_{BW,PLL}\) and damping ratio to \((k_p,k_i)\) is not stated.
- Eq. (14) has a numerator \(s\), so a fault voltage step produces a jump in \(\Delta\dot I_d\); the paper does not state that post-fault initial condition explicitly.
- The converter-side voltage transient used by the detailed simulation is omitted from Eq. (18).

The code therefore separates **equation-faithful dynamics** from a transparent **calibration layer** in `ReconstructionSettings`. The three calibration constants are chosen only to match the published anchor cases and the approximately 2-rad reduced-model first swing. They are not claimed to be the authors’ unpublished controller settings.

## Parameters

The physical values follow Table I:

- rated power: 7.35 kW;
- AC voltage: 400 V line-line RMS;
- DC voltage: 700 V;
- grid frequency: 50 Hz;
- filter inductance: 7.6 mH;
- grid inductance: 2.8 mH;
- active-current reference: 1 pu;
- reactive-current reference: 0 pu;
- fault voltage: 0.05 pu for the severe-fault cases.

The code uses the peak-value dq convention \(P=\frac{3}{2}V_dI_d\), giving \(I_{d,ref}\approx15.0\) A and pre-fault \(\delta_0\approx0.0404\) rad.

## Run

```bash
python -m pip install -r requirements.txt
python scripts/run_time_domain.py
python scripts/reconstruct_bandwidth_region.py
```

Outputs are written to `outputs/`.

## File map

- `src/model.py`: models, gains, initial conditions, integration and stability test.
- `scripts/run_time_domain.py`: frequency, current and EAC figures.
- `scripts/reconstruct_bandwidth_region.py`: Fig. 9 reconstruction.
- `data/fig9_digitized_boundary.csv`: approximate visual digitization of Fig. 9.

## Recommended next step

For a stricter reproduction, obtain either the authors’ Simulink model or the missing \(k_{pc},k_{ic}\) design rule. Then set all calibration factors to 1, replace `current_loop_gains`, and identify the exact fault-side initial jump from the detailed converter model.
