import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.force.eam_force_cu import EAMForceCUDAExt
from io_utils.eam_parser import EAMParser
from io_utils.reader import AtomFileReader
from io_utils.w_bcc import generate_oriented_bcc_w, write_xyz


EV_ANG3_TO_GPA = 160.21766208
EV_ANG3_TO_BAR = EV_ANG3_TO_GPA * 10_000.0


def _set_box_strain(mol, h0, frac0, strain_matrix):
    mol.box.H = strain_matrix.to(h0.dtype) @ h0
    coords = (frac0 @ mol.box.H.to(frac0.dtype)).to(mol.coordinates.dtype)
    mol.update_coordinates(coords)


def _measure(force, mol):
    out = force()
    sigma_bar = out["virial_tensor"].diag().to(torch.float64) / float(mol.box.volume) * EV_ANG3_TO_BAR
    return float(out["energy"]), sigma_bar


def main():
    root = Path(__file__).resolve().parents[1]
    eam_path = root / "run_data" / "W" / "WRe_YC2.eam.fs"
    tmp_xyz = root / "run_output" / "test_w_elastic_static" / "W_4x4x4.xyz"
    tmp_xyz.parent.mkdir(parents=True, exist_ok=True)

    parser = EAMParser(str(eam_path), device=torch.device("cpu"))
    coords, box_vectors = generate_oriented_bcc_w(
        lattice_param=3.1652,
        orientation="100",
        replicas=(4, 4, 4),
    )
    write_xyz(tmp_xyz, coords, atom_type="W", comment="W static elastic regression")

    mol = AtomFileReader(
        filename=str(tmp_xyz),
        box_length=float(torch.norm(box_vectors[0]).item()),
        cutoff=parser.cutoff,
        device=torch.device("cpu"),
        skin_thickness=1.0,
        is_mlp=True,
        box_vectors=box_vectors,
    )
    force = EAMForceCUDAExt(parser, mol, n_r=10_000, n_rho=10_000, use_extension=False)

    h0 = mol.box.H.detach().clone()
    frac0 = mol.coordinates.detach().clone().to(h0.dtype) @ torch.linalg.inv(h0)
    volume0 = float(mol.box.volume)
    eps = 1.0e-3

    identity = torch.eye(3, dtype=h0.dtype)
    e0, sigma0 = _measure(force, mol)

    plus = identity.clone()
    plus[0, 0] = 1.0 + eps
    _set_box_strain(mol, h0, frac0, plus)
    e_plus, sigma_plus = _measure(force, mol)

    minus = identity.clone()
    minus[0, 0] = 1.0 - eps
    _set_box_strain(mol, h0, frac0, minus)
    e_minus, sigma_minus = _measure(force, mol)

    c11_energy_gpa = ((e_plus + e_minus - 2.0 * e0) / (volume0 * eps * eps)) * EV_ANG3_TO_GPA
    c11_stress_gpa = -float(sigma_plus[0] - sigma_minus[0]) / (2.0 * eps) / 10_000.0
    c12_stress_gpa = -float(sigma_plus[1] - sigma_minus[1]) / (2.0 * eps) / 10_000.0
    rel_diff = abs(c11_energy_gpa - c11_stress_gpa) / max(abs(c11_energy_gpa), 1.0)

    print(
        "W static elastic: "
        f"C11_energy={c11_energy_gpa:.2f} GPa, "
        f"C11_stress={c11_stress_gpa:.2f} GPa, "
        f"C12_stress={c12_stress_gpa:.2f} GPa, "
        f"sigma0_bar={sigma0.detach().cpu().tolist()}"
    )

    assert 350.0 < c11_energy_gpa < 700.0, c11_energy_gpa
    assert 350.0 < c11_stress_gpa < 700.0, c11_stress_gpa
    assert 100.0 < c12_stress_gpa < 350.0, c12_stress_gpa
    assert rel_diff < 0.08, (c11_energy_gpa, c11_stress_gpa, rel_diff)
    print("W static elastic regression passed.")


if __name__ == "__main__":
    main()
