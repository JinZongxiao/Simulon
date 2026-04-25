# Changelog

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
