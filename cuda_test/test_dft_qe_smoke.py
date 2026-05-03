import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from postprocess.dft_qe import parse_qe_output
from run_scripts.run_dft_qe_task import _build_parser, run_qe_task


def _write_minimal_w_qe_input(path: Path, pseudo_dir: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "&CONTROL",
                "  calculation = 'scf'",
                "  prefix = 'w_bcc_qe_smoke'",
                f"  pseudo_dir = '{pseudo_dir}'",
                "  outdir = './tmp'",
                "  tstress = .true.",
                "  tprnfor = .true.",
                "/",
                "&SYSTEM",
                "  ibrav = 0",
                "  nat = 2",
                "  ntyp = 1",
                "  ecutwfc = 30",
                "  ecutrho = 240",
                "  occupations = 'smearing'",
                "  smearing = 'mv'",
                "  degauss = 0.02",
                "/",
                "&ELECTRONS",
                "  conv_thr = 1.0d-6",
                "/",
                "ATOMIC_SPECIES",
                "W 183.84 W_pbe_v1.2.uspp.F.UPF",
                "CELL_PARAMETERS angstrom",
                "3.1652 0.0000 0.0000",
                "0.0000 3.1652 0.0000",
                "0.0000 0.0000 3.1652",
                "ATOMIC_POSITIONS angstrom",
                "W 0.0000 0.0000 0.0000",
                "W 1.5826 1.5826 1.5826",
                "K_POINTS gamma",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _parser_only_smoke(tmp_dir: Path) -> None:
    qe_in = tmp_dir / "qe" / "pw.in"
    qe_out = tmp_dir / "qe" / "qe.out"
    _write_minimal_w_qe_input(qe_in, "/tmp/pseudo")
    qe_out.write_text(
        """
!    total energy              =    -300.00000000 Ry
     convergence has been achieved in   5 iterations
     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    2 type  1   force =     0.00000000    0.00000000    0.00000000

          total   stress  (Ry/bohr**3)                   (kbar)     P=    0.00
   0.00000000   0.00000000   0.00000000       0.00      0.00      0.00
   0.00000000   0.00000000   0.00000000       0.00      0.00      0.00
   0.00000000   0.00000000   0.00000000       0.00      0.00      0.00
   JOB DONE.
""",
        encoding="utf-8",
    )
    label = parse_qe_output(qe_out, qe_in)
    assert label["label_ready"] is True
    assert label["n_atoms"] == 2
    assert len(label["forces_eV_A"]) == 2
    assert len(label["stress_GPa"]) == 3


def main():
    output_dir = Path(__file__).resolve().parents[1] / "run_output" / "smoke_dft_qe_task"
    _parser_only_smoke(output_dir / "parser_only")

    pw = shutil.which("pw.x")
    mpirun = shutil.which("mpirun")
    pseudo_dir = os.environ.get("ESPRESSO_PSEUDO", "/public/home/normal_bgd/J1N/software/pseudopotentials")
    pseudo = Path(pseudo_dir) / "W_pbe_v1.2.uspp.F.UPF"
    if not pw or not mpirun or not pseudo.exists():
        print("QE executable or W pseudopotential not available; parser-only DFT QE smoke passed.")
        return

    task_dir = output_dir / "w_bcc_qe_smoke"
    qe_in = task_dir / "qe" / "pw.in"
    _write_minimal_w_qe_input(qe_in, pseudo_dir)
    parser = _build_parser()
    args = parser.parse_args(
        [
            str(task_dir),
            "--np",
            "2",
            "--omp",
            "1",
            "--timeout",
            "120",
        ]
    )
    status = run_qe_task(args)
    assert status["returncode"] == 0
    assert status["label_ready"] is True
    label_path = Path(status["label_json"])
    assert label_path.exists()
    with open(label_path, "r", encoding="utf-8") as f:
        label = json.load(f)
    assert label["backend"] == "qe"
    assert label["task_id"] == "w_bcc_qe_smoke"
    assert label["converged"] is True
    assert label["job_done"] is True
    assert label["energy_eV"] is not None
    assert len(label["forces_eV_A"]) == 2
    assert len(label["stress_GPa"]) == 3
    print("DFT QE smoke test passed.")


if __name__ == "__main__":
    main()
