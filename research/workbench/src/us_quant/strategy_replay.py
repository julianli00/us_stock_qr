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
    "growth_factor_satellite": "config/growth-factor-satellite.json",
    "conditional_factor_model": "config/conditional-factor-model.json",
    "factor_momentum_comparison": "config/factor-momentum-comparison.json",
    "trend_factor_guard": "config/trend-factor-guard.json",
    "momentum_implementation": "config/momentum-implementation.json",
    "credit_factor_guard": "config/credit-factor-guard.json",
    "six_factor_strategy": "config/six-factor-strategy.json",
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
            "cash_funded_accounting_v2": file_digest(directory / "cash_funded_accounting_v2.py"),
            "factor_family_sources": file_digest(directory / "factor_family_sources.py"),
        }
    )


def registered_generator(spec: dict, root: Path) -> tuple[str, dict]:
    frozen = spec.get("frozen_files", {})
    matches = []
    for name, policy_path in GENERATORS.items():
        source = f"src/us_quant/{name}.py"
        if source not in frozen or policy_path not in frozen:
            continue
        required = [source, policy_path]
        if name == "factor_momentum_comparison":
            required += [
                "src/us_quant/factor_gold_risk.py",
                "config/factor-gold-risk.json",
                "src/us_quant/factor_research.py",
            ]
        elif name == "trend_factor_guard":
            required += [
                "src/us_quant/factor_gold_risk.py",
                "config/factor-gold-risk.json",
                "src/us_quant/multifactor_stability.py",
            ]
        elif name == "momentum_implementation":
            required += [
                "src/us_quant/factor_gold_risk.py",
                "config/factor-gold-risk.json",
                "src/us_quant/factor_replication.py",
                "src/us_quant/multifactor_stability.py",
            ]
        elif name == "credit_factor_guard":
            required += [
                "src/us_quant/adaptive_factor_allocation.py",
                "config/adaptive-factor-allocation.json",
                "src/us_quant/macro_factor_tilt.py",
                "data/credit-spread-source-20261011/acquisition.json",
                "data/credit-spread-source-20261011/BAA10Y.csv",
                "data/credit-spread-source-20261011/BAA10Y-source.html",
                "data/credit-spread-source-20261011/ICE-source-restriction.html",
            ]
        elif name == "six_factor_strategy":
            required += [
                "src/us_quant/factor_gold_risk.py",
                "config/factor-gold-risk.json",
                "src/us_quant/factor_family_sources.py",
                "config/factor-family-expansion.json",
                "src/us_quant/multifactor_stability.py",
                "src/us_quant/factor_replication.py",
                "data/new-factor-family-source-20261011/manifest.json",
            ]
        if any(relative not in frozen for relative in required):
            raise QuantError("The registered strategy is missing a frozen helper dependency.")
        for relative in required:
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
        elif name == "growth_factor_satellite":
            from us_quant.growth_factor_satellite import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "conditional_factor_model":
            from us_quant.conditional_factor_model import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "factor_momentum_comparison":
            from us_quant.factor_momentum_comparison import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "trend_factor_guard":
            from us_quant.trend_factor_guard import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "momentum_implementation":
            from us_quant.momentum_implementation import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "credit_factor_guard":
            from us_quant.credit_factor_guard import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "six_factor_strategy":
            from us_quant.six_factor_strategy import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
    return seed_window(cache[key], start)
