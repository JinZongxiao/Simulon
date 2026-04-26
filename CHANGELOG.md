# Changelog

## 2026-04-26

### Added

- Added a full W nanoindentation protocol with loading, optional hold, and
  unloading phases. The workflow now writes `nanoindent_log.csv`,
  `load_depth.png`, `load_depth_with_popin.png`, `summary.json`, `report.md`,
  `trajectory.xyz`, `snapshots/`, and `snapshots_png/`.
- Added nanoindentation pop-in detection from load drops or sudden loading
  stiffness drops, with the detected point annotated in
  `load_depth_with_popin.png` when present.
- Added geometric spherical-contact hardness reporting for indentation via
  `A = pi(2Rh - h^2)` and `hardness_method=geometric_spherical_contact_area`.
- Added mechanism classification for W crack runs. `summary.json` now reports
  `classification`, `classification_reason`, `crack_opening_pass`,
  `significant_crack_propagation_pass`, `physics_acceptance_pass`, and explicit
  plasticity availability fields.
- Added crack-growth diagnostics to `run_scripts/w_crack.py`, including
  `crack_length_A`, `crack_extension_A`, key-strain XYZ/PNG snapshots, and
  threshold-sensitivity outputs for crack-length tracking.
- Added `run_scripts/w_crack_sweep.py` for crack-propagation acceptance sweeps
  over temperature, crack length, target strain, grip thickness, and run length.
- Added `run_scripts/rebuild_w_dbtt_summary.py` to rebuild DBTT summaries from
  existing per-temperature `summary.json` files without rerunning MD.
- Added DBTT mechanism summary plotting via `dbtt_mechanism_summary.png`.

### Changed

- W indentation summaries now report complete loading/unloading analysis fields:
  `max_depth_A`, `max_load_nN`, `residual_depth_A`,
  `unloading_stiffness_nN_per_A`, `work_loading`, `work_unloading`,
  `plastic_work_fraction`, `contact_area_A2`, `hardness_GPa`,
  `pop_in_detected`, and explicit plasticity-proxy availability.
- DBTT summaries now use per-temperature classifications instead of treating the
  first opening-acceptance pass as a DBTT candidate.
- Uniform `opening_only` scans now report `dbtt_physics_pass=false`,
  `dbtt_status=not_identified`, and `dbtt_candidate_temperature_k=null`.
- Existing W31250 100-800 K crack-opening scans are now interpreted as workflow
  passes but DBTT physics not identified unless a brittle-to-ductile mechanism
  contrast is present.

### Validation

- Server-side checks on `comput5` passed for `test_w_crack_smoke.py`,
  `test_w_dbtt_smoke.py`, and `w_crack_sweep.py --smoke`.
- A W31250 low-temperature sweep case reached significant-propagation acceptance
  (`max_crack_extension_A > 2 A`, `max_CMOD_A > 3 A`, and post-peak stress drop),
  while stronger opening cases remained correctly classified as `opening_only`
  when long-range crack extension was not observed.

## 2026-04-24

### Fixed

- Corrected stress conversion from `eV/A^3` to `bar`. The correct factor is
  `1 eV/A^3 = 1,602,176.6208 bar`; the old value was lower by a factor of 10
  and made W stress-strain slopes appear 10x too soft.
- Updated W barostat compressibility defaults to `3.2e-7 bar^-1`, matching the
  corrected pressure units and the expected bulk-modulus scale for W.
- Corrected LAMMPS `eam/fs` parsing in `EAMParser`.
  `WRe_YC2.eam.fs` is a Finnis-Sinclair potential with density tables indexed by
  both host and neighbor element, `rho[host, neighbor, r]`. It was previously
  parsed like `eam/alloy`, which shifted the pair-potential table and made W
  appear to have a false zero-pressure lattice constant near `3.34 A`.
- Updated the EAM force path to use host-neighbor density tables for both energy
  density accumulation and force derivatives. A regression now checks that pure W
  at the potential-file lattice constant `a = 3.1652 A` has near-zero virial stress.
- Fixed the NPT barostat sign convention used by W equilibration, including both
  the anisotropic tensile barostat and the isotropic bulk-relax Berendsen path.
  Positive virial stress in this code path is compression-positive, so the
  barostat must expand positive-stress axes and contract negative-stress axes.
  The old sign could shrink an already-compressed box and drive pre-load stresses
  to tens or hundreds of kbar.
- Made `tension_*` columns and `sigma_xx_tension` logging genuinely
  tensile-positive. Native `stress_*` columns keep the virial convention, while
  `tension_*` columns are sign-flipped for plotting and interpretation.
- Changed generated W workflow defaults from `a = 3.2 A` to `a = 3.1652 A`,
  matching the bundled `WRe_YC2.eam.fs` W lattice parameter.

### CUDA

- Updated the CUDA EAM extension kernels to accept density and density-derivative
  tables shaped `[host_element, neighbor_element, r]`, matching the Python
  Finnis-Sinclair path.
- Server-side CUDA users should rebuild after pulling:

  ```bash
  python setup.py build_ext --inplace
  ```

### Validation

- Added a regression check that pure W at `a = 3.1652 A` evaluates to near-zero
  virial stress with the bundled `WRe_YC2.eam.fs` potential.
- Added a W tensile smoke check with a real 6x6x6 pre-equilibrated cell and
  tensile-positive output sign validation.
- Existing `run_output` directories are local runtime artifacts and should be
  regenerated after pulling if old plots or summaries were produced before this
  fix.
