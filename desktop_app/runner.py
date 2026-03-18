from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, TextIO

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = Path.home() / "Documents" / "Simulon" / "output"

ROOT_IMPORT_PATH = str(ROOT_DIR)
if ROOT_IMPORT_PATH not in sys.path:
    sys.path.insert(0, ROOT_IMPORT_PATH)

from io_utils.output_logger import Logger
from io_utils.reader import AtomFileReader
import torch
from core.force.lennard_jones_force import LennardJonesForce
from core.force.template.pair_force_template import PairForce
from core.integrator.integrator import VerletIntegrator
from core.md_model import BaseModel, SumBackboneInterface
from core.md_simulation import MDSimulator


@contextlib.contextmanager
def redirected_output(stdout_stream: TextIO | None, stderr_stream: TextIO | None = None) -> Iterator[None]:
    if stdout_stream is None and stderr_stream is None:
        yield
        return
    stdout_backup = sys.stdout
    stderr_backup = sys.stderr
    sys.stdout = stdout_stream or stdout_backup
    sys.stderr = stderr_stream or stderr_backup
    try:
        yield
    finally:
        sys.stdout = stdout_backup
        sys.stderr = stderr_backup


def _normalize_path(raw_path: str, config_dir: Path) -> str:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        candidate = (config_dir / path).resolve()
        if candidate.exists():
            return str(candidate)
        fallback = (ROOT_DIR / raw_path).resolve()
        return str(fallback)
    return str(path)


def _prepare_config(config: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    payload = json.loads(json.dumps(config))
    for key in ("data_path_xyz", "aimd_pos_file", "aimd_force_file"):
        if key in payload and payload[key]:
            payload[key] = _normalize_path(payload[key], config_dir)

    output_dir = payload.get("output_save_path") or str(DEFAULT_OUTPUT_DIR)
    output_path = Path(output_dir).expanduser()
    if not output_path.is_absolute():
        output_path = (DEFAULT_OUTPUT_DIR / output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    payload["output_save_path"] = str(output_path)
    return payload


class TeeStream:
    def __init__(self, primary: TextIO, secondary: TextIO | None = None) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, s: str) -> int:
        self.primary.write(s)
        if self.secondary is not None:
            self.secondary.write(s)
        return len(s)

    def flush(self) -> None:
        self.primary.flush()
        if self.secondary is not None:
            self.secondary.flush()


def run_simulation(mode: str, config: dict[str, Any], config_dir: Path | None = None, stream: TextIO | None = None) -> dict[str, str]:
    config_dir = config_dir or ROOT_DIR
    payload = _prepare_config(config, config_dir)
    output_save_path = payload["output_save_path"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tee_stdout = TeeStream(Logger(sys.__stdout__, log_dir=output_save_path), stream)
    tee_stderr = TeeStream(Logger(sys.__stderr__, log_dir=output_save_path), stream)

    with redirected_output(tee_stdout, tee_stderr):
        print(f"Resolved input config relative to: {config_dir}")
        print(f"Simulation mode: {mode}")
        print(f"Simulation will use device: {device}")
        atom_file_reader = AtomFileReader(
            payload["data_path_xyz"],
            box_length=payload["box_length"],
            cutoff=payload["cut_off"],
            device=device,
            parameter=payload.get("pair_parameter"),
            skin_thickness=3.0,
            is_mlp=(mode == "mlps"),
        )

        if mode == "lj":
            force_field = LennardJonesForce(atom_file_reader)
        elif mode == "user_defined":
            pair_parameter = payload.get("pair_parameter") or {}
            if not pair_parameter:
                raise ValueError("pair_parameter is required for user_defined mode")
            parameter_names = list(next(iter(pair_parameter.values())).keys())
            force_field = PairForce(atom_file_reader, parameter_names, payload["potential_formula"])
        else:
            raise ValueError(f"Unsupported desktop mode: {mode}")

        combined_field = SumBackboneInterface([force_field], atom_file_reader)
        integrator = VerletIntegrator(
            atom_file_reader,
            payload["dt"],
            force_field,
            "NVT",
            payload["temperature"],
            payload["gamma"],
        )
        model = BaseModel(combined_field, integrator, atom_file_reader)
        simulation = MDSimulator(model, payload["num_steps"], payload["print_interval"], save_to_graph_dataset=False)
        simulation.run(enable_minimize_energy=(mode != "user_defined"))

        now_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        energy_path = os.path.join(output_save_path, f"MD_energy_curve_{now_time}.png")
        simulation.save_energy_curve(energy_path)
        traj_path = os.path.join(output_save_path, f"MD_traj_{now_time}.xyz")
        simulation.save_xyz_trajectory(traj_path, atom_types=atom_file_reader.atom_types)
        force_path = os.path.join(output_save_path, f"forces_{now_time}.xyz")
        simulation.save_forces_grad(force_path, with_no_ele=True, atom_types=atom_file_reader.atom_types)
        print(f"Saved energy curve to: {energy_path}")
        print(f"Saved trajectory to: {traj_path}")
        print(f"Saved forces to: {force_path}")

    return {
        "output_dir": output_save_path,
        "energy_curve": energy_path,
        "trajectory": traj_path,
        "forces": force_path,
    }
