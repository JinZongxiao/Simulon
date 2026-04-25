import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.force.eam_force_cu import EAMForceCUDAExt
from io_utils.eam_parser import EAMParser
from io_utils.reader import AtomFileReader
from io_utils.w_bcc import generate_oriented_bcc_w, write_xyz


EV_ANG3_TO_BAR = 1_602_176.6208


def main():
    root = Path(__file__).resolve().parents[1]
    eam_path = root / "run_data" / "W" / "WRe_YC2.eam.fs"
    tmp_xyz = root / "run_output" / "test_eam_fs_parser" / "W_3p1652.xyz"
    tmp_xyz.parent.mkdir(parents=True, exist_ok=True)

    parser = EAMParser(str(eam_path), device=torch.device("cpu"))
    assert parser.is_fs_format, "WRe_YC2.eam.fs must be parsed as Finnis-Sinclair"
    assert parser.density_splines_by_pair[0][0] is not parser.density_splines_by_pair[0][1]

    coords, box_vectors = generate_oriented_bcc_w(
        lattice_param=3.1652,
        orientation="100",
        replicas=(4, 4, 4),
    )
    write_xyz(tmp_xyz, coords, atom_type="W", comment="W FS parser regression")
    mol = AtomFileReader(
        filename=str(tmp_xyz),
        box_length=float(torch.norm(box_vectors[0]).item()),
        cutoff=parser.cutoff,
        device=torch.device("cpu"),
        skin_thickness=1.0,
        is_mlp=True,
        box_vectors=box_vectors,
    )
    out = EAMForceCUDAExt(parser, mol, use_extension=False)()
    stress_diag = out["virial_tensor"].diag().to(torch.float64) / float(mol.box.volume)
    stress_diag_bar = stress_diag * EV_ANG3_TO_BAR
    assert float(torch.max(torch.abs(stress_diag_bar))) < 20.0, stress_diag_bar.tolist()
    print("EAM FS parser regression passed.")


if __name__ == "__main__":
    main()
