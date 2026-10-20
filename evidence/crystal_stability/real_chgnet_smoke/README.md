# Actual initial-geometry to formation-energy smoke

The script reads only the final task input: the first valid test structure (`crystal_000001`, 8 atoms), the 89 elemental references and the frozen CHGNet checkpoint. Python socket connection methods are disabled before scientific-library imports. It does not read withheld stability labels or prediction archives.

CHGNet 0.3.0 performs genuine E/F/S inference, then ASE FIRE with FrechetCellFilter updates both atomic coordinates and the cell for 10 steps. The initial model energy is -5.324212551116943 eV/atom; the final energy is -5.339659214019775 eV/atom. The maximum generalized filter force decreases from 0.5213238678 to 0.3750583131. The 0.01 convergence threshold **is not reached**. Cell and Cartesian-position Frobenius changes are 0.1227792488 angstrom and 0.2466538713 angstrom respectively.

Subtracting the supplied stoichiometric elemental reference (-5.913733758125 eV/atom) gives the finite demonstration formation energy **0.5740745441052244 eV/atom**. No additional MP2020 compatibility correction is applied in this bounded smoke. This is not an accuracy claim, a converged relaxation, or a fresh run of all 215,488 structures. The one-row CSV is a demonstration, not a complete valid benchmark submission.

`smoke_result.json` records every source input hash, selected row, full E/F/S output, relaxation parameters, all 11 trajectory snapshots, actual convergence state, formation-energy arithmetic, versions and timing. `run.log` and `relaxation.log` preserve the actual run. Calculation/setup time inside main() is 0.296 seconds on two CPU threads; Python import/startup time is excluded. This execution used the tested macOS research environment, not a Docker/Linux execution.

Reproduce from the bundle root after installing the research environment:

```sh
/opt/sie-research/bin/python evidence/crystal_stability/run_chgnet_relaxation_smoke.py --input tasks/crystal_stability/input --output evidence/crystal_stability/real_chgnet_smoke
```
