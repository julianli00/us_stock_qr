from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_horizon import seed_window
from us_quant.storage import digest_json, file_digest, implementation_fingerprint, read_json

GENERATORS = {
    "factor_gold_risk": "config/factor-gold-risk.json",
    "factor_replication": "config/factor-implementation-replication.json",
    "adaptive_factor_allocation": "config/adaptive-factor-allocation.json",
    "volatility_term_risk": "config/volatility-term-risk.json",
    "macro_factor_tilt": "config/macro-factor-tilt.json",
    "defensive_factor_rotation": "config/defensive-factor-rotation.json",
}


def replay_dependencies_hash() -> str:
    directory = Path(__file__).resolve().parent
    return digest_json(
        {
            "replay": file_digest(Path(__file__)),
            "core": implementation_fingerprint(),
            "independent_accounting": file_digest(directory / "bt_audit.py"),
            "independent_metrics": file_digest(directory / "factor_validation.py"),
            "window_logic": file_digest(directory / "dual_horizon.py"),
        }
    )


def registered_generator(spec: dict, root: Path) -> tuple[str, dict]:
    frozen = spec.get("frozen_files", {})
    matches = []
    for name, policy_path in GENERATORS.items():
        source = f"src/us_quant/{name}.py"
        if source not in frozen or policy_path not in frozen:
            continue
        for relative in (source, policy_path):
            path = root / relative
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or not path.is_file()
                or file_digest(path) != frozen[relative]
            ):
                raise QuantError("The registered strategy implementation or policy was changed.")
        executable = Path(__file__).resolve().parent / f"{name}.py"
        if file_digest(executable) != frozen[source]:
            raise QuantError("The executable registered strategy is not the frozen source file.")
        policy = read_json(root / policy_path)
        candidate = next(
            (item for item in policy.get("candidates", []) if item.get("id") == spec.get("id")),
            None,
        )
        if candidate is not None:
            if candidate != spec.get("configuration"):
                raise QuantError("The registered strategy configuration differs from its policy.")
            matches.append((name, policy))
    if len(matches) != 1:
        raise QuantError(
            "No unique registered strategy target generator; submitted curves cannot qualify."
        )
    return matches[0]


def registered_targets(
    spec: dict,
    data: MarketData,
    start: str,
    end: str,
    cost: float,
    delay: int,
    root: Path,
    cache: dict,
) -> pd.DataFrame:
    name, policy = registered_generator(spec, root)
    candidate = spec["configuration"]
    key = (name, digest_json(spec), data.close.index[0], data.close.index[-1])
    if name == "factor_gold_risk":
        from us_quant.factor_gold_risk import monthly_targets, run_candidate

        if key not in cache:
            cache[key] = monthly_targets(data, policy)
        _, targets, _ = run_candidate(data, cache[key], candidate, policy, start, end, cost, delay)
        return targets
    if name == "factor_replication":
        from us_quant.factor_replication import run

        return run(data, candidate, policy, start, end, cost, delay)[1]
    if key not in cache:
        if name == "adaptive_factor_allocation":
            from us_quant.adaptive_factor_allocation import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "volatility_term_risk":
            from us_quant.volatility_term_risk import SOURCES, build_targets, load_terms

            cache[key] = build_targets(
                data, load_terms(SOURCES, data.close.index), candidate, policy
            )
        elif name == "macro_factor_tilt":
            from us_quant.macro_factor_tilt import SOURCE, build_targets, load_macro

            macro, _ = load_macro(SOURCE, data.close.index)
            cache[key] = build_targets(data, macro, candidate, policy)
        elif name == "defensive_factor_rotation":
            from us_quant.defensive_factor_rotation import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
    return seed_window(cache[key], start)
