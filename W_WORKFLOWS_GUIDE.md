# W Workflows Guide

## Scope

This guide documents the pure-W structure builder, the production pure-W structure baseline matrix, the four independent pure-W mechanics workflows in Simulon, plus one auxiliary bulk-relax preparation workflow:

- `run_scripts/build_w_structure.py`
- `run_scripts/w_structure_baseline.py`
- `run_scripts/w_gb_search.py`
- `run_scripts/w_bulk_relax.py`
- `run_scripts/w_tensile.py`
- `run_scripts/w_indent.py`
- `run_scripts/w_crack.py`
- `run_scripts/w_dbtt_scan.py`

Each workflow can be run alone. The batch wrapper `run_scripts/w_batch_report.py` only orchestrates the four mechanics workflows; it does not currently include `w_bulk_relax.py`.

## Output Structure

Every workflow accepts `--output-dir`. Outputs are grouped by orientation underneath that root:

- tensile: `.../orientation_100/`, `.../orientation_110/`, `.../orientation_111/`
- indentation: `.../orientation_100/`, `.../orientation_110/`, `.../orientation_111/`
- crack: `.../orientation_100/`, `.../orientation_110/`, `.../orientation_111/`
- dbtt: `.../orientation_100/`, `.../orientation_110/`, `.../orientation_111/` with temperature subdirectories inside
- bulk relax: `.../orientation_100/`, `.../orientation_110/`, `.../orientation_111/`
- structure builder: `.../<case_name>/`
- structure baseline: `.../cases/<case_name>/`, plus `structure_baseline.csv`, `summary.json`, and `report.md`
- grain-boundary search: `.../<case_name>/`

When `--orientation custom` is used, the same layout becomes `.../orientation_custom/`.

This layout is intentional so you can run one workflow, two workflows, or all workflows without file collisions.

## Pure W Structure Builder

`run_scripts/build_w_structure.py` prepares geometry-only W input structures. It is useful before mechanics runs and before later ODS-W embedding work.

Supported first-stage kinds:

- `bulk`
- `surface`
- `vacancy`
- `interstitial`
- `substitution`
- `void`
- `bicrystal`
- `crack`
- `notch`

Example commands:

```bash
python run_scripts/build_w_structure.py --kind bulk --orientation 100 --replicas 10,10,10
python run_scripts/build_w_structure.py --kind surface --orientation 110 --replicas 10,10,6 --vacuum-A 30
python run_scripts/build_w_structure.py --kind vacancy --orientation 100 --replicas 10,10,10 --vacancy-count 5
python run_scripts/build_w_structure.py --kind void --orientation 100 --replicas 12,12,12 --void-radius-A 8
python run_scripts/build_w_structure.py --kind bicrystal --gb-plane 3,1,0 --replicas 8,6,6
python run_scripts/build_w_structure.py --kind crack --orientation 100 --replicas 20,10,10 --crack-half-length-A 30 --crack-opening-A 2
python run_scripts/build_w_structure.py --kind notch --orientation 100 --replicas 20,10,10 --notch-radius-A 10 --notch-depth-A 10
```

Use `--relax` to run fixed-box steepest-descent geometry relaxation immediately after building:

```bash
python run_scripts/build_w_structure.py \
  --kind vacancy \
  --orientation 100 \
  --replicas 8,8,8 \
  --vacancy-count 2 \
  --relax \
  --relax-steps 500 \
  --relax-force-threshold 0.05
```

Each case writes:

- `structure.xyz`
- `summary.json`
- `composition.csv`
- `preview.png`
- optional `relaxed_structure.xyz`
- optional `relaxation.csv`
- optional `relax_summary.json`

`bicrystal` currently means a CSL-periodic BCC `[001]` symmetric tilt grain-boundary seed. The default `--gb-plane 3,1,0` is `Sigma5(310)[001]` with misorientation `36.8699 deg`; `--gb-plane 2,1,0` gives the other common `Sigma5(210)[001]` branch with misorientation `53.1301 deg`. The summary records `sigma`, `misorientation_deg`, `gb_plane_hkl`, `tilt_axis_uvw`, grain atom counts, and whether the construction is CSL-exact.

Important limitation: builder relaxation is fixed-box steepest descent. It is intended to remove severe local forces after construction, not to replace production NVT/NPT relaxation. Grain-boundary production work still needs rigid-body translation search and relaxation. Dislocations are deliberately deferred because they need a dedicated elastic displacement field and core validation.

## Production Pure-W Structure Baseline Matrix

Script: `run_scripts/w_structure_baseline.py`

Purpose: build and fixed-box relax a reproducible pure-W geometry baseline library before ODS-W embedding or defect-mechanics comparisons.

Production command:

```bash
python run_scripts/w_structure_baseline.py \
  --preset production \
  --orientation 100 \
  --relax-method fire \
  --relax-steps 3000 \
  --relax-force-threshold 0.05 \
  --output-dir run_output/prod_w_structure_baseline
```

The production preset includes:

- `bulk_100`: periodic bulk W reference
- `surface_100_z`: free-surface geometry seed with vacuum along z
- `vacancy_1`: single-vacancy W
- `interstitial_1`: single W interstitial seed
- `void_r8`: spherical void seed
- `crack_seed`: precrack geometry seed
- `notch_seed`: surface-notch geometry seed
- `bicrystal_seed_sigma5_310_001`: raw CSL seed
- `gb_search_sigma5_310_001`: rigid-body translation searched GB candidate

Main outputs:

- `structure_baseline.csv`
  One row per case, with atom count, energy, force, structure paths, and validation flags.
- `summary.json`
  Machine-readable workflow summary and per-case records.
- `report.md`
  Human-readable baseline report.
- `cases/<case_name>/`
  Per-case structure outputs from `build_w_structure.py`.

Important fields:

- `acceptance_pass`
  The case built successfully, files exist, no NaN/Inf energy was produced, and relaxation did not increase energy.
- `production_ready`
  Stronger than `acceptance_pass`. For ordinary structures, the final maximum force must satisfy `--relax-force-threshold`; for GB search, the GB-energy sanity criteria must also pass.
- `relax_force_pass`
  Whether the final fixed-box relaxation force is below threshold.
- `--relax-method`
  Engineering meaning: fixed-box relaxation algorithm. `fire` is recommended for production defect seeds; `sd` preserves the original steepest-descent path for simple smoke/debug runs.

Smoke test:

```bash
python cuda_test/test_w_structure_baseline_smoke.py
```

Smoke output is for API/output validation only. It intentionally uses very short relaxation and may mark some cases as not `production_ready` even though `acceptance_pass=true`.

## Grain-Boundary Rigid-Body Translation Search

Script: `run_scripts/w_gb_search.py`

Purpose: start from the strict CSL bicrystal construction, scan rigid-body translations of one grain in the GB plane, relax every candidate with fixed-box steepest descent, and report the lowest-energy GB candidate.

Default geometry:

- `--gb-plane 3,1,0`: `Sigma5(310)[001]` BCC W symmetric tilt boundary
- two periodic GBs in the simulation cell
- translation grid in the GB plane: `x` and `z`

Example:

```bash
python run_scripts/w_gb_search.py \
  --gb-plane 3,1,0 \
  --replicas 8,6,6 \
  --translations-x 5 \
  --translations-z 3 \
  --relax-steps 500 \
  --relax-force-threshold 0.05 \
  --output-dir run_output/w_gb_search
```

Production-style starting point:

```bash
python run_scripts/w_gb_search.py \
  --gb-plane 3,1,0 \
  --replicas 12,8,8 \
  --translations-x 7 \
  --translations-z 5 \
  --gb-overlap-cutoff-A 1.6 \
  --gb-search-width-A 6.0 \
  --relax-steps 1000 \
  --relax-force-threshold 0.05 \
  --output-dir run_output/prod_w_gb_search_sigma5_310_001
```

Key parameters:

- `--gb-plane`
  Physical meaning: CSL boundary plane `(h,k,0)` for a BCC `[001]` symmetric tilt boundary. `h` and `k` must be coprime positive integers.
- `--translations-x`, `--translations-z`
  Physical meaning: rigid-body translation grid counts along the two in-plane directions. Larger grids are more expensive but reduce the chance of selecting a bad microscopic GB state.
- `--gb-overlap-cutoff-A`
  Physical meaning: close-pair removal distance near the GB planes before relaxation.
- `--gb-search-width-A`
  Engineering meaning: spatial window around each periodic GB plane used for overlap cleanup.
- `--bulk-energy-per-atom-ev`
  Physical meaning: bulk W reference energy used in the GB excess-energy formula. Default is `auto`, which evaluates a BCC W bulk reference with the same EAM file and lattice parameter. Prefer `auto` unless you intentionally want to compare against an external reference.
- `--bulk-reference-replicas`
  Engineering meaning: optional bulk reference supercell for `auto`; defaults to the same `--replicas`.
- `--relax-steps`, `--relax-step-size-A`, `--relax-force-threshold`
  Engineering meaning: fixed-box steepest-descent relaxation controls for each candidate.

Outputs:

- `candidates.csv`
  One row per translation candidate, including shift, atom count, relaxed energy, energy per atom, and final force.
- `best_structure.xyz`
  The lowest-energy unrelaxed candidate.
- `best_relaxed_structure.xyz`
  The lowest-energy relaxed candidate for downstream production relaxation.
- `gb_energy_report.json`
  Machine-readable GB-energy report.
- `summary.json`
  Workflow-level summary with the embedded `best` report.

`gb_energy_report.json` fields:

- `bulk_reference`
  Records whether the bulk reference was `auto` or user-supplied. For `auto`, it includes the reference structure, atom count, total energy, and eV/atom.
- `gb_energy_J_m2`
  Excess grain-boundary energy in J/m^2:
  `(E_GB - N * E_bulk_per_atom) / (2 * A_GB)`.
- `gb_energy_valid`
  Conservative sanity flag requiring positive GB energy and a small energy-per-atom offset from the bulk reference. If this is false, do not use the candidate for production.
- `csl_exact`
  Whether the bicrystal construction used the explicit CSL geometry path.

Smoke test:

```bash
python cuda_test/test_w_gb_search_smoke.py
```

## Common Parameters

These appear in more than one workflow.

- `--orientation`
  Physical meaning: crystal orientation used to generate the W BCC cell. Supported values are `100`, `110`, `111`, and `custom`.
- `--structure`
  Engineering meaning: path to an input XYZ file. This is only used when `--orientation custom`.
- `--box-length`
  Physical meaning: cubic box length in Angstrom for `--orientation custom`.
  Current implementation assumes the imported XYZ coordinates belong to an orthogonal cubic periodic cell.
  If your custom structure is not cubic, do not use this path yet.
- `--replicas`
  Physical meaning: supercell size along the three lattice vectors of the oriented cell. Larger values reduce size effects and boundary artifacts, but cost more GPU memory and wall time.
  This is ignored when `--orientation custom`.
- `--eam`
  Physical meaning: EAM parameter file used for W interactions.
- `--temperature`
  Physical meaning: target thermostat temperature in K.
- `--dt`
  Physical meaning: MD time step in ps.
- `--gamma`
  Physical meaning: Langevin damping in `1/ps`. Larger values mean stronger thermostat coupling.
- `--output-dir`
  Engineering meaning: root directory where this workflow writes CSV, summary JSON, PNG, and generated structure files.
- `--smoke`
  Engineering meaning: small acceptance run. It is for code-path validation, not for publishable physics.

## Custom Large-Structure Mode

All four workflows support `--orientation custom`.

This mode is intended for server runs on larger W structures that already exist as XYZ files, for example:

- `run_data/W/W250.xyz`
- `run_data/W/W31250.xyz`

For the current implementation, the imported structure must satisfy all of the following:

- the XYZ contains Cartesian coordinates only
- the true simulation box is an orthogonal cubic box
- you pass that cubic edge length through `--box-length`
- the structure is bulk-like and periodic before the workflow adds vacuum or a crack

Example: `W31250.xyz`

- first line: `31250` atoms
- BCC W has 2 atoms per conventional cell
- `31250 / 2 = 15625 = 25^3`
- lattice parameter is `3.2 A`
- cubic box length is `25 x 3.2 = 80.0 A`

So the correct custom arguments for this file are:

- `--orientation custom`
- `--structure run_data/W/W31250.xyz`
- `--box-length 80.0`

## Bulk Relax Parameters

Script: `run_scripts/w_bulk_relax.py`

Purpose: relax a bulk W cell toward zero pressure before using it in a tensile run.

- `--target-pressure-bar`
  Physical meaning: target isotropic pressure for bulk relaxation.
- `--barostat-tau`
  Physical meaning: Berendsen pressure relaxation time.
- `--barostat-compressibility-bar-inv`
  Physical meaning: effective isotropic compressibility used by the Berendsen barostat. For W, the default `3.2e-7 bar^-1` is consistent with a few-hundred-GPa bulk modulus scale.
- `--barostat-mu-max`
  Engineering meaning: maximum isotropic scaling per step. Lower values are slower but safer for large systems.

### Bulk Relax Report Fields

- `recommended_box_length_A`
  Mean final box length after relaxation. Use this as the next `--box-length` for `--orientation custom` tensile runs.
- `recommended_lattice_param_A`
  Estimated relaxed cubic BCC lattice parameter when the script can infer the number of cubic cells per axis from atom count.
- `final_pressure_bar`
  Final mean pressure after relaxation.
- `final_box_length_x/y/z`
  Final box lengths written into the relaxed structure.

### Bulk Relax Large-Structure Example

```bash
python run_scripts/w_bulk_relax.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --steps 5000 \
  --temperature 300 \
  --gamma 2.0 \
  --target-pressure-bar 0.0 \
  --barostat-tau 0.5 \
  --barostat-compressibility-bar-inv 3.2e-7 \
  --barostat-mu-max 0.005 \
  --traj-interval 500 \
  --output-dir run_output/w_bulk_relax_W31250
```

Use the relaxed XYZ plus `recommended_box_length_A` as the input for the next tensile attempt.

## Tensile Parameters

Script: `run_scripts/w_tensile.py`

- `--strain-rate`
  Physical meaning: engineering strain rate in `1/ps` along the loading axis.
- `--lateral-mode`
  Physical meaning:
  `fixed`: lateral box lengths remain fixed.
  `poisson`: lateral box lengths shrink with a prescribed Poisson ratio.
  `stress-free`: lateral directions are controlled by anisotropic NPT.
- `--poisson-ratio`
  Physical meaning: kinematic lateral contraction ratio for `poisson` mode.
- `--barostat-tau`
  Physical meaning: pressure relaxation time for anisotropic NPT. Smaller values respond faster but can destabilize the run.
- `--barostat-gamma`
  Physical meaning: damping applied to the barostat degrees of freedom.
- `--barostat-compressibility-bar-inv`
  Physical meaning: effective lateral compressibility used by the anisotropic pressure controller. This sets how strongly the lateral box reacts to a stress mismatch; for W, start from `3.2e-7 bar^-1`.
- `--barostat-pressure-tolerance-bar`
  Physical meaning: deadband around the target lateral stress. Inside this tolerance, the controller only damps its own rate instead of continuing to drift.
- `--max-lateral-box-ratio`
  Engineering meaning: safety cutoff on lateral box expansion relative to the equilibrated start of loading. If exceeded, the run aborts instead of silently writing a nonphysical curve.

### Tensile Report Fields

- `stress_max_bar`
  Peak axial stress, using the tension-positive convention.
- `peak_strain`
  Strain at peak axial stress.
- `elastic_slope_bar`
  Early-stage slope of the tension-positive axial response. This is a quick stiffness proxy, not a rigorous elastic constant fit.
- `final_stress_bar`
  Final axial stress, using the tension-positive convention.
- `stress_drop_bar`
  Difference between peak stress and final stress. Useful for identifying post-peak softening.
- `mean_final_lateral_stress_bar`
  Mean lateral stress near the end. Useful to judge how well `stress-free` loading released transverse stress.
- `stress_sign_convention`
  Current value: `tension_positive`.
- `recommended_plot_column`
  Current value: `tension_xx_bar`. This is the default and only column for the main tensile stress-strain plot.
- `lateral_stress_columns`
  Current value: `["tension_yy_bar", "tension_zz_bar"]`. These are transverse stress diagnostics for the stress-free barostat, not components to average into the tensile curve.
- `native_stress_sign_convention`
  Explains that `stress_*` columns retain the internal compression-positive virial sign and should not be used directly for presentation tensile curves.

The tensile CSV contains both signed virial-style stress columns (`stress_*`) and tension-positive presentation columns (`tension_*`). Use `tension_xx_bar` / `tension_bar` for plotting and interpretation. Do not average `xx`, `yy`, and `zz`: uniaxial tensile stress is the axial component. The generated `stress_strain.png`, `summary.json`, and `report.md` follow this tension-positive convention, while `lateral_stress.png` is only a barostat diagnostic.

For large `--orientation custom` runs, also inspect `initial_stress_xx_abs_bar`, `initial_stress_yy_abs_bar`, and `initial_stress_zz_abs_bar` in `summary.json`. If they remain large after equilibration, extend `--equil-steps` or retune the barostat before interpreting the tensile response.

### Tensile Large-Structure Example

```bash
python run_scripts/w_tensile.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --steps 100000 \
  --equil-steps 1000 \
  --strain-rate 0.0004 \
  --lateral-mode stress-free \
  --barostat-tau 0.1 \
  --barostat-gamma 1.0 \
  --barostat-compressibility-bar-inv 3.2e-7 \
  --barostat-pressure-tolerance-bar 25.0 \
  --max-lateral-box-ratio 2.0 \
  --gamma 2.0 \
  --traj-interval 1000 \
  --output-dir run_output/prod_w_tensile_W31250
```

## Indentation Parameters

Script: `run_scripts/w_indent.py`

- `--vacuum-A`
  Physical meaning: extra vacuum added normal to the free surface to avoid periodic-image contact above the slab.
- `--bottom-thickness-A`
  Physical meaning: thickness of the rigid bottom grip region. These atoms are held fixed.
- `--equil-steps`
  Physical meaning: NVT equilibration steps before indentation begins.
- `--indenter-radius-A`
  Physical meaning: spherical indenter radius in Angstrom.
- `--indenter-stiffness`
  Physical meaning: repulsive indenter stiffness in `eV/A^3`. Larger values approach a harder indenter.
- `--initial-depth-A`
  Physical meaning: initial effective indentation depth relative to the geometric contact reference. `0.0` means start at geometric first contact.
- `--target-depth-A`
  Physical meaning: target effective indentation depth.
- `--hold-steps`
  Physical meaning: optional constant-depth hold steps after loading.
- `--hold-depth-A`
  Physical meaning: constant hold depth. If omitted, the target depth is used.
- `--unload-steps`
  Physical meaning: unloading steps after loading/hold. The default enables an unloading segment.
- `--final-unload-depth-A`
  Physical meaning: final command depth at the end of unloading.
- `--indent-rate-A-ps`
  Physical meaning: imposed indentation speed in `A/ps`. If omitted, it is inferred from `(target-depth - initial-depth) / (steps * dt)`.
- `--unload-rate-A-ps`
  Physical meaning: imposed unloading speed in `A/ps`. If omitted, it is inferred from `(hold-depth - final-unload-depth) / (unload-steps * dt)`.
- `--traj-interval`
  Physical meaning: optional interval for additional trajectory frames. `trajectory.xyz` is always written with at least initial/final states.

### Indentation Report Fields

- `nanoindent_log.csv`
  Complete loading/hold/unloading log. Required fields include `step`, `time_ps`, `phase`, `depth_A`, `load_nN`, `indenter_z`, `temp`, `pot`, `kin`, and `total`.
- `report.md`
  Human-readable production report with system, protocol, main results, interpretation, and limitations.
- `max_load_nN`
  Maximum load during the run.
- `max_depth_A`
  Maximum indentation depth.
- `residual_depth_A`
  Residual depth estimated from the initial unloading stiffness intercept when unloading exists, otherwise the final logged depth.
- `peak_load_depth_A`
  Depth where the load reaches its maximum.
- `unloading_stiffness_nN_per_A`
  Linear slope from the first unloading points, in `nN/A`.
- `work_loading`
  Trapezoidal loading work in `nN*A`.
- `work_unloading`
  Trapezoidal recovered unloading work in `nN*A`.
- `plastic_work_fraction`
  `(work_loading - work_unloading) / work_loading`, clipped to `[0, 1]`.
- `contact_area_A2`
  Geometric spherical contact area `A = pi(2Rh - h^2)`.
- `hardness_GPa`
  `Pmax / contact_area`, using `1 nN/A^2 = 100 GPa`.
- `hardness_method`
  Currently `geometric_spherical_contact_area`.
- `pop_in_detected`, `pop_in_depth_A`, `pop_in_load_nN`
  Pop-in detection from load drops or sudden loading-stiffness drops. If no event is detected, `pop_in_detected=false`.
- `plasticity_indicator_available`
  Currently `false` unless a real CSP/non-bcc/displacement proxy is implemented.
- `max_contact_atoms`
  Maximum number of atoms simultaneously inside the indenter interaction zone.

Generated outputs include `nanoindent_log.csv`, legacy `load_depth.csv`, `load_depth.png`, `load_depth_with_popin.png`, `summary.json`, `report.md`, `trajectory.xyz`, `snapshots/`, and `snapshots_png/`.

The hardness field is currently a geometric workflow-level estimate. It is useful for comparing runs inside Simulon, but it should not yet be treated as a fully calibrated experimental nanoindentation pipeline.

### Indentation Large-Structure Example

Use a bulk custom structure. The workflow will add vacuum normal to the indentation direction and create the free surface internally.

```bash
python run_scripts/w_indent.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --steps 10000 \
  --equil-steps 1000 \
  --hold-steps 1000 \
  --unload-steps 5000 \
  --indenter-radius-A 8.0 \
  --indenter-stiffness 5.0 \
  --initial-depth-A 0.0 \
  --target-depth-A 4.0 \
  --final-unload-depth-A 0.5 \
  --gamma 2.0 \
  --traj-interval 500 \
  --output-dir run_output/prod_w_indent_W31250
```

## Crack Parameters

Script: `run_scripts/w_crack.py`

- `--vacuum-A`
  Physical meaning: extra vacuum normal to the opening direction to avoid interactions across the free surfaces.
- `--crack-half-length-A`
  Physical meaning: half-length of the initial center crack.
- `--crack-gap-A`
  Physical meaning: opening thickness of the removed precrack strip.
- `--grip-thickness-A`
  Physical meaning: thickness of the rigid upper and lower grip regions used for displacement control.
- `--equil-steps`
  Physical meaning: NVT equilibration steps before the crack is opened.
- `--target-strain`
  Physical meaning: prescribed remote opening strain based on the gauge region.
- `--opening-rate-A-ps`
  Physical meaning: imposed crack-mouth opening rate in `A/ps`. If omitted, it is inferred from `target opening / (steps * dt)`.
- `--crack-open-threshold-A`
  Physical meaning: local upper-lower crack-plane separation used to classify a bin as open for crack-length tracking.
- `--crack-length-bins`
  Engineering meaning: number of bins along the crack direction used for estimated crack-length tracking.

### Crack Report Fields

- `stress_bar`
  Opening stress using the tension-positive convention. The raw internal compression-positive virial sign is also written as `native_stress_yy_bar`.
- `stress_drop_ratio`
  `(peak_tensile_stress_bar - final_stress_bar) / peak_tensile_stress_bar`. Values above about `0.1` indicate visible post-peak load drop.
- `peak_stress_magnitude_bar`
  Peak opening tensile stress during the run. Kept under this historical field name for DBTT compatibility.
- `peak_stress_at_final_step`
  Boolean flag for whether the peak stress occurs at the final recorded point. If true, the case has not yet shown post-peak unloading.
- `cmod_at_peak_stress_A`
  Crack-mouth opening displacement at the stress peak.
- `max_cmod_A`
  Largest crack-mouth opening displacement reached.
- `max_crack_length_A`
  Estimated length of the connected opened crack region.
- `max_crack_extension_A`
  Estimated crack extension beyond the initial slit length.
- `stress_at_max_cmod_bar`
  Stress when the maximum CMOD occurs.
- `initial_cmod_slope_A_per_strain`
  Early CMOD-versus-strain slope. This is a compliance proxy.
- `stress_retention_ratio`
  Final stress divided by peak stress magnitude.
- `fracture_work_proxy_bar_A`
  Area under the stress-CMOD response. Use it as a relative fracture-work proxy, not as a direct toughness value.
- `classification`
  Mechanism label: `brittle`, `ductile`, `opening_only`, `no_crack_growth`, or `invalid`.
- `classification_reason`
  Short explanation for the mechanism label.
- `crack_opening_pass`
  True when CMOD, stress drop, and peak-step criteria show a real opening response.
- `significant_crack_propagation_pass`
  True when crack extension is at least `2 A`, CMOD is at least `3 A`, stress drop is at least `0.15`, and peak stress is not at the final step.
- `physics_acceptance_pass`
  True only when the response can be interpreted as `brittle`, `ductile`, or `opening_only`.
- `plasticity_indicator_available`
  Currently `false`; no DXA/CSP/non-affine plasticity proxy is used yet.
- `crack_tracking_reliable`
  Reliability flag from recalculating crack extension with thresholds `0.5, 0.8, 1.0, 1.2, 1.5 A`.

### Crack Large-Structure Example

Use a bulk custom structure. The workflow will add opening-direction vacuum and cut the initial center crack internally.

```bash
python run_scripts/w_crack.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --steps 10000 \
  --equil-steps 1000 \
  --crack-half-length-A 8.0 \
  --crack-gap-A 1.2 \
  --target-strain 0.03 \
  --gamma 2.0 \
  --traj-interval 500 \
  --output-dir run_output/prod_w_crack_W31250
```

## DBTT Scan Parameters

Script: `run_scripts/w_dbtt_scan.py`

This workflow repeatedly calls the crack workflow at multiple temperatures.

- `--temperatures`
  Physical meaning: comma-separated temperature list in K.
- `--temperature-scale`
  Engineering meaning: multiplier applied to every listed temperature. Useful when reusing a list under a systematic scaling study.
- `--steps`, `--equil-steps`, `--dt`, `--gamma`
  Same meanings as in the crack workflow.
- `--crack-half-length-A`, `--crack-gap-A`, `--grip-thickness-A`, `--target-strain`, `--opening-rate-A-ps`
  Same meanings as in the crack workflow.
- `--crack-open-threshold-A`, `--crack-length-bins`, `--traj-interval`, `--print-interval`
  Same meanings as in the crack workflow.

### DBTT Report Fields

- `classification`
  Per-temperature crack mechanism label.
- `dbtt_workflow_pass`
  True when all per-temperature jobs finished and the combined summary was generated.
- `dbtt_physics_pass`
  True only when the scan shows a conservative brittle-to-ductile mechanism contrast. It remains false for uniform opening-only scans.
- `dbtt_status`
  `candidate_identified`, `not_identified`, or `insufficient_data`.
- `dbtt_candidate_temperature_k`
  Candidate transition temperature. This is `null` unless a mechanism contrast is present.
- `classification_counts`
  Count of `brittle`, `ductile`, `opening_only`, `no_crack_growth`, and `invalid` cases.
- `peak_stress_magnitude_bar`
  Temperature dependence of peak opening-stress magnitude. Keep it for reference, but do not use it alone to identify the transition.
- `max_cmod_A`
  Temperature dependence of maximum crack opening.
- `cmod_at_peak_stress_A`
  Crack opening when the crack workflow reaches peak stress. Useful as a simple brittleness-versus-ductility proxy.
- `final_stress_bar`
  Residual load carrying capacity at the end of the crack-opening path.
- `stress_retention_ratio`
  Final stress divided by peak stress magnitude. Lower values indicate stronger post-peak softening.

For the current crack-based W DBTT workflow, do not interpret a uniform `opening_only` scan as DBTT. The workflow can pass while DBTT physics remains `not_identified`.

## Crack Propagation Sweep

Script: `run_scripts/w_crack_sweep.py`

Default recommended case:

```bash
python run_scripts/w_crack_sweep.py \
  --orientation custom \
  --structure run_output/prod_w_bulk_relax_W31250/orientation_custom/W_custom_relaxed.xyz \
  --box-length 79.28473306554223
```

Supported grid parameters:

- `--temperatures`
- `--crack-half-lengths-A`
- `--target-strains`
- `--grip-thicknesses-A`
- `--steps-list`
- `--equil-steps-list`

The default targets `T=100 K`, `crack_half_length_A=28`, `target_strain=0.10`, `steps=15000`, `equil_steps=2000`, and `grip_thickness_A=5`.

### DBTT Large-Structure Example

```bash
python run_scripts/w_dbtt_scan.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --temperatures 100,200,300,400,500,600 \
  --steps 5000 \
  --equil-steps 500 \
  --gamma 2.0 \
  --output-dir run_output/prod_w_dbtt_W31250
```

## Batch Runner

Script: `run_scripts/w_batch_report.py`

Purpose: run any subset of the four workflows and produce a combined report.

Key parameters:

- `--workflows`
  Example: `tensile,indent,crack,dbtt`
- `--orientations`
  Example: `100,110,111` or `custom`
- `--output-dir`
  Root directory for all selected workflow outputs and the combined report.
- `--replicas-100`, `--replicas-110`, `--replicas-111`
  Orientation-specific supercell sizes shared by all selected workflows.
- `--structure`, `--box-length`
  Shared custom-structure inputs passed through to every selected workflow when `--orientations custom`.

### Batch Report Meaning

The batch runner writes:

- `batch_report.csv`
  Flat table of key metrics, suitable for spreadsheets or quick filtering.
- `batch_report.json`
  Machine-readable version of the same information.
- `batch_report.md`
  Human-readable run index with the main metrics and file paths.

This report is not a paper-ready analysis by itself. It is intended to organize large server sweeps and let you quickly identify which runs need closer inspection.

### Batch Large-Structure Example

```bash
python run_scripts/w_batch_report.py \
  --workflows tensile,indent,crack,dbtt \
  --orientations custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --output-dir run_output/w_batch_W31250
```

## Practical Server Advice

- Start with `--smoke` locally.
- Before `W31250.xyz`, do one short server sanity run with the same file, for example 200 to 500 steps.
- Increase `--replicas` before increasing loading rate if you see strong size artifacts.
- Keep `--dt` conservative when using high strain rate, high opening rate, or a stiff indenter.
- For DBTT or crack studies, use multiple temperatures and, eventually, multiple random seeds if you need trends robust enough for reporting.
- Do not run several heavy CUDA workflows on the same GPU at the same time unless you are deliberately stress-testing throughput. For shallow smoke tests, concurrent GPU jobs can make acceptance behavior look noisy.
