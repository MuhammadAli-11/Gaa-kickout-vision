"""Config loading + deterministic run identity.

Every stage resolves its parameters here. The run_id is a hash of the
config subset that affects the result, so two runs with different
thresholds can never overwrite each other's outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(p)
    return cfg


def _stable_hash(obj: Any, n: int = 8) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def run_id(cfg: dict, stages: tuple[str, ...] = ("video", "detect", "track", "features", "events")) -> str:
    """Deterministic id from the parameters that change the numbers."""
    subset = {k: cfg.get(k) for k in stages}
    return f"run_{_stable_hash(subset)}"


def out_dir(cfg: dict, create: bool = True) -> Path:
    d = REPO_ROOT / cfg["paths"]["outputs"] / run_id(cfg)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] to an absolute path."""
    return REPO_ROOT / cfg["paths"][key]


def set_seeds(cfg: dict) -> None:
    seed = int(cfg["project"]["seed"])
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "nogit"


@dataclass
class Provenance:
    """Written beside every artefact. Answers 'what produced this number?'"""
    stage: str
    run_id: str
    git_sha: str
    config_hash: str
    created_utc: str
    inputs: list[str]
    params: dict

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.__dict__, fh, indent=2, default=str)


def make_provenance(stage: str, cfg: dict, inputs: list[str], params: dict | None = None) -> Provenance:
    from datetime import datetime, timezone
    return Provenance(
        stage=stage,
        run_id=run_id(cfg),
        git_sha=git_sha(),
        config_hash=_stable_hash(cfg),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        inputs=inputs,
        params=params or {},
    )
