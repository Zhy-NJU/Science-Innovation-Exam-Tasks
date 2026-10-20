#!/usr/bin/env python3
"""Real, offline CHGNet -> FC2/FC3 -> Wigner-RTA conductivity smoke test.

Reads only supplied crystal geometries and the supplied model checkpoint, never
test labels or archived prediction tables. The smaller supercell/q-mesh are a
workflow smoke configuration, not a benchmark-accuracy or convergence claim.
"""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import torch
from chgnet.model.model import CHGNet
from phono3py import Phono3py
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.core import Structure


def block_network(*args, **kwargs):
    raise RuntimeError("Network access is disabled in this scientific smoke test")


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supercell", type=int, default=2)
    parser.add_argument("--qmesh", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    socket.create_connection = block_network
    socket.socket.connect = block_network
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    checkpoint = args.input / "model_weights/chgnet_0.3.0_e29f68s314m37.pth.tar"
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_sha == "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1"
    with gzip.open(args.input / "structures.jsonl.gz", "rt") as f:
        candidates = [json.loads(line) for line in f]
    candidates = [r for r in candidates if set(r["atomic_numbers"]) == {8, 12} and len(r["atomic_numbers"]) == 8]
    assert len(candidates) == 1, "Expected one supplied 8-atom rocksalt MgO cell"
    row = candidates[0]
    model = CHGNet.from_file(str(checkpoint), mlp_out_bias=False, version="0.3.0").to("cpu")
    model.eval()

    def predict(numbers, cell, positions, task="ef"):
        structure = Structure(lattice=cell, species=numbers, coords=positions, coords_are_cartesian=True)
        prediction = model.predict_structure(structure, task=task)
        clean = {k: np.asarray(v, dtype=np.float64) for k, v in prediction.items()}
        assert all(np.isfinite(v).all() for v in clean.values())
        assert clean["f"].shape == (len(numbers), 3)
        return clean

    positions = np.asarray(row["positions_angstrom"], dtype=float)
    initial = predict(row["atomic_numbers"], row["cell_angstrom"], positions, "efs")
    displacement = 0.03
    plus, minus = positions.copy(), positions.copy()
    plus[0, 0] += displacement
    minus[0, 0] -= displacement
    positive = predict(row["atomic_numbers"], row["cell_angstrom"], plus)
    negative = predict(row["atomic_numbers"], row["cell_angstrom"], minus)
    force_difference = positive["f"] - negative["f"]
    assert np.max(np.abs(force_difference)) > 1e-5
    self_stiffness = -force_difference[0, 0] / (2 * displacement)
    assert math.isfinite(self_stiffness) and self_stiffness > 0
    details = {
        "status": "efs_and_central_displacement_passed",
        "scientific_scope": "One supplied MgO geometry; real pretrained-model forces, real finite-displacement FC2/FC3 and real Wigner-RTA kappa. Coarse workflow smoke, not the 103-material benchmark result and not a converged physical prediction.",
        "material_id": row["material_id"], "chemical_system": "MgO rocksalt",
        "relaxed_during_smoke": False,
        "host": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()},
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "scipy", "chgnet", "pymatgen", "ase", "phonopy", "phono3py", "spglib", "symfc", "monty"]},
        "device": "cpu", "torch_threads": 2, "python_socket_connects": "disabled before model loading and all scientific calculations",
        "checkpoint_sha256": checkpoint_sha,
        "input_structure": row,
        "initial_prediction": initial,
        "plus_displacement_prediction": positive,
        "minus_displacement_prediction": negative,
        "central_displacement_angstrom": displacement,
        "central_self_stiffness_eV_per_A2": self_stiffness,
    }

    def save():
        (args.output / "smoke_result.json").write_text(json.dumps(details, indent=2, default=serialize, allow_nan=False) + "\n")

    save()
    print("EFS and +/- displacement forces passed", flush=True)
    cell = PhonopyAtoms(numbers=row["atomic_numbers"], cell=row["cell_angstrom"], positions=positions)
    ph3 = Phono3py(cell, supercell_matrix=[args.supercell] * 3,
                  primitive_matrix=row["primitive_matrix"], symprec=1e-5, log_level=1)
    ph3.nac_params = None
    ph3.generate_displacements(distance=displacement, is_plusminus=True)
    displaced_cells = ph3.supercells_with_displacements
    assert displaced_cells and all(sc is not None for sc in displaced_cells)
    raw_forces, raw_energies = [], []
    for index, sc in enumerate(displaced_cells):
        prediction = predict(sc.numbers.tolist(), sc.cell, sc.positions)
        forces = prediction["f"]
        # Remove only the uniform net-force drift of this model evaluation.
        corrected = forces - forces.mean(axis=0)
        raw_forces.append(corrected)
        raw_energies.append(float(prediction["e"]))
        if (index + 1) % 10 == 0 or index + 1 == len(displaced_cells):
            print(f"Model force calculations {index + 1}/{len(displaced_cells)}", flush=True)
    force_array = np.asarray(raw_forces, dtype=np.float64)
    assert np.isfinite(force_array).all()
    ph3.forces = force_array
    ph3.produce_fc3(symmetrize_fc3r=True)
    ph3.produce_fc2(symmetrize_fc2=True)
    assert np.isfinite(ph3.fc2).all() and np.isfinite(ph3.fc3).all()
    np.savez_compressed(args.output / "smoke_force_constants.npz", fc2=ph3.fc2, fc3=ph3.fc3,
                        displaced_forces=force_array, displaced_energies_eV_per_atom=raw_energies)
    details["force_constants"] = {"supercell_matrix": [args.supercell] * 3,
                                  "supercell_atoms": len(displaced_cells[0]),
                                  "displaced_supercell_count": len(displaced_cells),
                                  "force_drift_removed": True,
                                  "displacement_angstrom": displacement,
                                  "fc2_shape": list(ph3.fc2.shape), "fc3_shape": list(ph3.fc3.shape),
                                  "fc2_unit": "eV/angstrom^2", "fc3_unit": "eV/angstrom^3",
                                  "fc2_max_abs": float(np.max(np.abs(ph3.fc2))),
                                  "fc3_max_abs": float(np.max(np.abs(ph3.fc3))),
                                  "fc2_symmetrization": True, "fc3_symmetrization": True}
    details["status"] = "fc2_fc3_passed"
    save()
    ph3.mesh_numbers = [args.qmesh] * 3
    ph3.init_phph_interaction()
    ph3.run_phonon_solver()  # Explicitly calculate every point used by the stability check.
    ph3.run_thermal_conductivity(temperatures=[300.0], is_LBTE=False,
                                is_isotope=True, transport_type="MS-SMM19", write_kappa=False)
    thermal = ph3.thermal_conductivity
    totals = np.asarray(thermal.kappa_TOT_RTA, dtype=np.float64)
    particles = np.asarray(thermal.kappa_P_RTA, dtype=np.float64)
    coherence = np.asarray(thermal.kappa_C, dtype=np.float64)
    assert totals.shape == particles.shape == coherence.shape == (1, 1, 6)
    assert np.isfinite(totals).all() and np.isfinite(particles).all() and np.isfinite(coherence).all()
    kappa = float(totals[0, 0, :3].mean())
    assert kappa > 0
    assert np.allclose(totals, particles + coherence, rtol=1e-10, atol=1e-10)
    frequencies = ph3.get_phonon_data()[0]
    assert np.isfinite(frequencies).all()
    minimum_frequency = float(np.min(frequencies))
    assert minimum_frequency > -1e-3, f"Nontrivial imaginary phonons: {minimum_frequency} THz"
    details["thermal_conductivity"] = {
        "temperature_K": 300.0, "q_mesh": [args.qmesh] * 3,
        "no_NAC": True, "Wigner_transport": True, "phono3py_transport_type": "MS-SMM19", "single_mode_RTA": True,
        "natural_isotope_scattering": True,
        "kappa_TOT_RTA_W_mK": totals, "kappa_P_RTA_W_mK": particles, "kappa_C_W_mK": coherence,
        "direction_average_total_kappa_W_mK": kappa,
        "minimum_phonon_frequency_THz": minimum_frequency,
        "benchmark_comparison_performed": False,
        "convergence_claim": False,
        "benchmark_settings_difference": "Smoke uses FC2/FC3 2x2x2 conventional supercell and 5x5x5 q-mesh by default; the supplied benchmark MgO settings use FC2 4x4x4, FC3 2x2x2, and q-mesh 19x19x19."
    }
    details["status"] = "passed_real_efs_fc2_fc3_wigner_rta"
    details["wall_seconds"] = time.monotonic() - start
    save()
    print(json.dumps({"status": details["status"], "kappa_W_mK": kappa,
                      "wall_seconds": details["wall_seconds"], "minimum_frequency_THz": minimum_frequency}), flush=True)


if __name__ == "__main__":
    main()
