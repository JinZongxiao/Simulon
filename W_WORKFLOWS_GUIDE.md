# W Workflows Guide

## Scope

This guide documents the four independent pure-W mechanics workflows in Simulon, plus one auxiliary bulk-relax preparation workflow:

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

When `--orientation custom` is used, the same layout becomes `.../orientation_custom/`.

This layout is intentional so you can run one workflow, two workflows, or all workflows without file collisions.

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

The tensile CSV contains both signed virial-style stress columns (`stress_*`) and tension-positive presentation columns (`tension_*`). Use the `tension_*` columns for plotting and interpretation.

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
- `--indent-rate-A-ps`
  Physical meaning: imposed indentation speed in `A/ps`. If omitted, it is inferred from `(target-depth - initial-depth) / (steps * dt)`.

### Indentation Report Fields

- `max_load_nN`
  Maximum load during the run.
- `peak_load_depth_A`
  Depth where the load reaches its maximum.
- `contact_onset_depth_A`
  First depth where the load becomes nonzero.
- `initial_loading_stiffness_nN_per_A`
  Early loading slope after contact.
- `max_contact_atoms`
  Maximum number of atoms simultaneously inside the indenter interaction zone.
- `hardness_GPa`
  Oliver-Pharr-style hardness estimate from the current loading-unloading cycle.
- `reduced_modulus_GPa`
  Oliver-Pharr-style reduced modulus estimate from the initial unloading stiffness.

These hardness and modulus fields are currently workflow-level estimates. They are useful for comparing runs inside Simulon, but they should not yet be treated as a fully calibrated experimental nanoindentation pipeline.

### Indentation Large-Structure Example

Use a bulk custom structure. The workflow will add vacuum normal to the indentation direction and create the free surface internally.

```bash
python run_scripts/w_indent.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --steps 10000 \
  --equil-steps 1000 \
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

### Crack Report Fields

- `stress_bar`
  Opening stress using the tension-positive convention. The raw internal compression-positive virial sign is also written as `native_stress_yy_bar`.
- `peak_stress_magnitude_bar`
  Peak opening tensile stress during the run. Kept under this historical field name for DBTT compatibility.
- `cmod_at_peak_stress_A`
  Crack-mouth opening displacement at the stress peak.
- `max_cmod_A`
  Largest crack-mouth opening displacement reached.
- `stress_at_max_cmod_bar`
  Stress when the maximum CMOD occurs.
- `initial_cmod_slope_A_per_strain`
  Early CMOD-versus-strain slope. This is a compliance proxy.
- `stress_retention_ratio`
  Final stress divided by peak stress magnitude.
- `fracture_work_proxy_bar_A`
  Area under the stress-CMOD response. Use it as a relative fracture-work proxy, not as a direct toughness value.

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

### DBTT Report Fields

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

For the current crack-based W DBTT workflow, interpret the transition primarily through `final_stress_bar`, `stress_retention_ratio`, and `max_cmod_A`.

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
