from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from desktop_app.runner import DEFAULT_OUTPUT_DIR, run_simulation

APP_TITLE = "Simulon Desktop"
APP_DIR_NAME = "Simulon"


@dataclass(frozen=True)
class TemplateSpec:
    key: str
    label: str
    config_file: str
    mode: str


def bundled_root() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


TEMPLATES = {
    "lj": TemplateSpec("lj", "Lennard-Jones", "run_scripts/lj_run.json", "lj"),
    "user_defined": TemplateSpec("user_defined", "User Defined Pair Potential", "run_scripts/user_defined_run.json", "user_defined"),
}


class TkTextWriter:
    def __init__(self, output_queue: "queue.Queue[str]") -> None:
        self.output_queue = output_queue

    def write(self, text: str) -> None:
        if text:
            self.output_queue.put(text)

    def flush(self) -> None:
        return


class SimulonDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1100x760")
        self.root.minsize(960, 640)

        self.output_queue: "queue.Queue[str]" = queue.Queue()
        self.run_thread: threading.Thread | None = None
        self.active_config_path: Path | None = None
        self.current_template = TEMPLATES["lj"]
        self.last_result: dict[str, Any] | None = None

        self.template_var = tk.StringVar(value=self.current_template.key)
        self.status_var = tk.StringVar(value="Ready")
        self.output_dir_var = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))

        self._build_layout()
        self._load_template(self.current_template)
        self._poll_output_queue()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(container)
        controls.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(controls, text="Simulation Template:").pack(side=tk.LEFT)
        template_combo = ttk.Combobox(
            controls,
            state="readonly",
            width=28,
            textvariable=self.template_var,
            values=[spec.key for spec in TEMPLATES.values()],
        )
        template_combo.pack(side=tk.LEFT, padx=(8, 8))
        template_combo.bind("<<ComboboxSelected>>", self._on_template_changed)

        ttk.Button(controls, text="Load Template", command=self._handle_reload_template).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Open Config", command=self._handle_open_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Save Config As", command=self._handle_save_config_as).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Run", command=self._handle_run).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Open Output", command=self._open_output_dir).pack(side=tk.LEFT, padx=4)

        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.RIGHT)

        output_row = ttk.Frame(container)
        output_row.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(output_row, text="Output Directory:").pack(side=tk.LEFT)
        output_entry = ttk.Entry(output_row, textvariable=self.output_dir_var)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        ttk.Button(output_row, text="Browse", command=self._choose_output_dir).pack(side=tk.LEFT)

        panes = ttk.PanedWindow(container, orient=tk.VERTICAL)
        panes.pack(fill=tk.BOTH, expand=True)

        config_frame = ttk.Labelframe(panes, text="Simulation Config (JSON)")
        self.config_text = tk.Text(config_frame, wrap=tk.NONE, undo=True, font=("Courier New", 11))
        config_x = ttk.Scrollbar(config_frame, orient=tk.HORIZONTAL, command=self.config_text.xview)
        config_y = ttk.Scrollbar(config_frame, orient=tk.VERTICAL, command=self.config_text.yview)
        self.config_text.configure(xscrollcommand=config_x.set, yscrollcommand=config_y.set)
        self.config_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        config_y.pack(side=tk.RIGHT, fill=tk.Y)
        config_x.pack(side=tk.BOTTOM, fill=tk.X)
        panes.add(config_frame, weight=3)

        log_frame = ttk.Labelframe(panes, text="Run Log")
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED, font=("Courier New", 10))
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        panes.add(log_frame, weight=2)

    def _on_template_changed(self, _event: object | None = None) -> None:
        self.current_template = TEMPLATES[self.template_var.get()]
        self._load_template(self.current_template)

    def _handle_reload_template(self) -> None:
        self._load_template(self.current_template)

    def _load_template(self, template: TemplateSpec) -> None:
        config_path = bundled_root() / template.config_file
        payload = self._read_json_file(config_path)
        payload["output_save_path"] = str(DEFAULT_OUTPUT_DIR)
        self.active_config_path = config_path
        self._set_config(payload)
        self._append_log(f"Loaded template: {template.label}\n")
        self.status_var.set(f"Template ready: {template.label}")

    def _handle_open_config(self) -> None:
        filename = filedialog.askopenfilename(
            title="Open JSON config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return
        config_path = Path(filename)
        try:
            payload = self._read_json_file(config_path)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Unable to open config:\n{exc}")
            return
        if "output_save_path" not in payload:
            payload["output_save_path"] = self.output_dir_var.get()
        self.active_config_path = config_path
        self._set_config(payload)
        self._append_log(f"Opened config: {config_path}\n")
        self.status_var.set(f"Opened: {config_path.name}")

    def _handle_save_config_as(self) -> None:
        try:
            payload = self._get_config_payload()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Invalid JSON:\n{exc}")
            return
        filename = filedialog.asksaveasfilename(
            title="Save JSON config",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if not filename:
            return
        config_path = Path(filename)
        config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.active_config_path = config_path
        self.status_var.set(f"Saved: {config_path.name}")
        self._append_log(f"Saved config: {config_path}\n")

    def _choose_output_dir(self) -> None:
        dirname = filedialog.askdirectory(title="Choose output directory")
        if dirname:
            self.output_dir_var.set(dirname)
            try:
                payload = self._get_config_payload()
                payload["output_save_path"] = dirname
                self._set_config(payload)
            except Exception:
                pass

    def _handle_run(self) -> None:
        if self.run_thread and self.run_thread.is_alive():
            messagebox.showinfo(APP_TITLE, "A simulation is already running.")
            return
        try:
            payload = self._get_config_payload()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Invalid JSON:\n{exc}")
            return
        payload["output_save_path"] = self.output_dir_var.get().strip() or str(DEFAULT_OUTPUT_DIR)
        self._set_config(payload)
        config_dir = self.active_config_path.parent if self.active_config_path else bundled_root()
        template = self.current_template
        self.last_result = None
        self.status_var.set("Running…")
        self._append_log("\n=== Simulation started ===\n")
        self.run_thread = threading.Thread(
            target=self._run_in_background,
            args=(template.mode, payload, config_dir),
            daemon=True,
        )
        self.run_thread.start()

    def _run_in_background(self, mode: str, payload: dict[str, Any], config_dir: Path) -> None:
        writer = TkTextWriter(self.output_queue)
        try:
            result = run_simulation(mode=mode, config=payload, config_dir=config_dir, stream=writer)
            self.last_result = result
            self.output_queue.put("\n=== Simulation finished ===\n")
            for key, value in result.items():
                self.output_queue.put(f"{key}: {value}\n")
            self.root.after(0, lambda: self.status_var.set("Completed"))
        except Exception:
            self.output_queue.put(traceback.format_exc())
            self.root.after(0, lambda: self.status_var.set("Failed"))

    def _open_output_dir(self) -> None:
        target = Path(self.output_dir_var.get()).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", str(target)])
        elif os.name == "nt":
            os.startfile(str(target))
        else:
            subprocess.Popen(["xdg-open", str(target)])

    def _get_config_payload(self) -> dict[str, Any]:
        return json.loads(self.config_text.get("1.0", tk.END))

    def _set_config(self, payload: dict[str, Any]) -> None:
        self.config_text.delete("1.0", tk.END)
        self.config_text.insert(tk.END, json.dumps(payload, indent=2, ensure_ascii=False))

    def _read_json_file(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _poll_output_queue(self) -> None:
        try:
            while True:
                self._append_log(self.output_queue.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self._poll_output_queue)


def main() -> None:
    root = tk.Tk()
    try:
        root.iconname(APP_TITLE)
    except Exception:
        pass
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    SimulonDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
