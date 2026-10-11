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
    "growth_portfolio_protection": "config/growth-portfolio-protection.json",
    "downside_risk_balance": "config/downside-risk-balance.json",
    "sector_growth_balance": "config/sector-growth-balance.json",
    "dollar_risk_guard": "config/dollar-risk-guard.json",
    "financial_conditions_guard": "config/financial-conditions-guard.json",
    "sector_term_guard": "config/sector-term-guard.json",
    "inflation_factor_guard": "config/inflation-factor-guard.json",
    "claims_factor_guard": "config/claims-factor-guard.json",
    "factor_covariance_risk": "config/factor-covariance-risk.json",
    "sector_basket_strategy": "config/sector-basket-strategy.json",
    "semiconductor_portfolio_protection": "config/semiconductor-portfolio-protection.json",
    "joint_risk_confirmation": "config/joint-risk-confirmation.json",
    "semiconductor_volatility_target": "config/semiconductor-volatility-target.json",
    "semiconductor_trend_transfer": "config/semiconductor-trend-transfer.json",
    "residual_treasury_rotation": "config/residual-treasury-rotation.json",
    "sector_breadth_risk": "config/sector-breadth-risk.json",
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
        elif name == "growth_portfolio_protection":
            required += [
                "src/us_quant/growth_factor_satellite.py",
                "config/growth-factor-satellite.json",
                "src/us_quant/multifactor_stability.py",
                "src/us_quant/cash_funded_accounting_v2.py",
            ]
        elif name == "sector_growth_balance":
            required += [
                "src/us_quant/growth_factor_satellite.py",
                "config/growth-factor-satellite.json",
                "src/us_quant/multifactor_stability.py",
            ]
        elif name == "dollar_risk_guard":
            required += [
                "src/us_quant/sector_growth_balance.py",
                "config/sector-growth-balance.json",
                "src/us_quant/growth_factor_satellite.py",
                "config/growth-factor-satellite.json",
                "src/us_quant/multifactor_stability.py",
                "src/us_quant/macro_factor_tilt.py",
                "data/dollar-information-source-20261011/verified-manifest.json",
                "data/macro-rate-access-20261010/verified-manifest.json",
                "data/macro-rate-access-20261010/DFII10-curl-response.csv",
            ]
        elif name == "financial_conditions_guard":
            required += [
                "src/us_quant/dollar_risk_guard.py",
                "config/dollar-risk-guard.json",
                "src/us_quant/sector_growth_balance.py",
                "config/sector-growth-balance.json",
                "src/us_quant/growth_factor_satellite.py",
                "config/growth-factor-satellite.json",
                "src/us_quant/multifactor_stability.py",
                "data/financial-conditions-vintages-20261011/verified-manifest-v2.json",
            ]
        elif name == "sector_term_guard":
            required += [
                "src/us_quant/volatility_term_risk.py",
                "config/volatility-term-risk.json",
                "src/us_quant/dollar_risk_guard.py",
                "config/dollar-risk-guard.json",
                "src/us_quant/sector_growth_balance.py",
                "config/sector-growth-balance.json",
                "src/us_quant/growth_factor_satellite.py",
                "config/growth-factor-satellite.json",
                "src/us_quant/multifactor_stability.py",
                "data/sector-term-source-20261011/verified-manifest.json",
                "data/volatility-term-source-20261010/manifest.json",
                "data/volatility-term-source-20261010/VIX.csv",
                "data/volatility-term-source-20261010/VIX3M.csv",
            ]
        elif name == "inflation_factor_guard":
            required += [
                "src/us_quant/dollar_risk_guard.py",
                "config/dollar-risk-guard.json",
                "src/us_quant/sector_growth_balance.py",
                "config/sector-growth-balance.json",
                "src/us_quant/growth_factor_satellite.py",
                "config/growth-factor-satellite.json",
                "src/us_quant/multifactor_stability.py",
                "src/us_quant/macro_factor_tilt.py",
                "src/us_quant/financial_conditions_guard.py",
                "src/us_quant/inflation_inputs.py",
                "config/inflation-inputs.json",
                "data/dollar-information-source-20261011/verified-manifest.json",
                "data/inflation-inputs-20261011/manifest.json",
                "evidence/inflation_input_readiness_20261011.json",
            ]
        elif name == "claims_factor_guard":
            required += [
                "src/us_quant/dollar_risk_guard.py",
                "config/dollar-risk-guard.json",
                "src/us_quant/sector_growth_balance.py",
                "config/sector-growth-balance.json",
                "src/us_quant/growth_factor_satellite.py",
                "config/growth-factor-satellite.json",
                "src/us_quant/multifactor_stability.py",
                "src/us_quant/macro_factor_tilt.py",
                "src/us_quant/financial_conditions_guard.py",
                "src/us_quant/jobless_claims_inputs.py",
                "src/us_quant/inflation_inputs.py",
                "config/jobless-claims-inputs.json",
                "data/dollar-information-source-20261011/verified-manifest.json",
                "data/claims-inputs-20261011/manifest.json",
                "evidence/jobless_claims_input_readiness_20261011.json",
                "data/claims-confirmation-context-20261011/context-verification.json",
                "reports/claims-confirmation-plan-20261011/availability-rejected-policy-v1.json",
                "reports/claims-confirmation-plan-20261011/availability-rejected-implementation-v1.py",
                "reports/claims-confirmation-plan-20261011/source-window-failure-v1.json",
            ]
        elif name == "factor_covariance_risk":
            required += [
                "evidence/risk_only_covariance_20261011_source_context.json",
                "evidence/factor_gold_risk_20261010_registration.json",
            ]
        elif name == "sector_basket_strategy":
            required += [
                "src/us_quant/sector_basket_sources.py",
                "src/us_quant/sector_growth_balance.py",
                "src/us_quant/factor_research.py",
                "src/us_quant/multifactor_stability.py",
                "src/us_quant/dual_horizon.py",
                "src/us_quant/dollar_risk_guard.py",
                "src/us_quant/macro_factor_tilt.py",
                "config/sector-basket-sources.json",
                "config/factor-research.json",
                "config/dual-horizon.json",
                "config/dollar-risk-guard.json",
                "evidence/factor_round_20261007_registration.json",
                "evidence/factor_gold_risk_20261010_registration.json",
                "evidence/sector_basket_source_readiness_20261011.json",
                "data/sector-basket-sources-20261011/manifest.json",
                "data/dollar-information-source-20261011/verified-manifest.json",
            ]
        elif name == "semiconductor_portfolio_protection":
            required += [
                "src/us_quant/growth_portfolio_protection.py",
                "config/growth-portfolio-protection.json",
                "src/us_quant/cash_funded_accounting_v2.py",
                "src/us_quant/sector_growth_balance.py",
                "config/sector-growth-balance.json",
                "src/us_quant/growth_factor_satellite.py",
                "config/growth-factor-satellite.json",
                "src/us_quant/multifactor_stability.py",
                "src/us_quant/dollar_risk_guard.py",
                "config/dollar-risk-guard.json",
                "src/us_quant/macro_factor_tilt.py",
                "data/dollar-information-source-20261011/verified-manifest.json",
                "data/macro-rate-access-20261010/verified-manifest.json",
                "data/macro-rate-access-20261010/DFII10-curl-response.csv",
                "evidence/dollar_risk_guard_20261011_registration.json",
            ]
        elif name == "joint_risk_confirmation":
            required += [
                "src/us_quant/sector_term_guard.py", "config/sector-term-guard.json",
                "src/us_quant/sector_growth_balance.py", "config/sector-growth-balance.json",
                "src/us_quant/growth_factor_satellite.py", "config/growth-factor-satellite.json",
                "src/us_quant/multifactor_stability.py", "src/us_quant/dollar_risk_guard.py",
                "config/dollar-risk-guard.json", "src/us_quant/macro_factor_tilt.py",
                "src/us_quant/volatility_term_risk.py", "config/volatility-term-risk.json",
                "evidence/sector_term_guard_20261011_registration.json",
                "data/dollar-information-source-20261011/verified-manifest.json",
                "data/macro-rate-access-20261010/verified-manifest.json",
                "data/macro-rate-access-20261010/DFII10-curl-response.csv",
                "data/sector-term-source-20261011/verified-manifest.json",
                "data/volatility-term-source-20261010/manifest.json",
                "data/volatility-term-source-20261010/VIX.csv",
                "data/volatility-term-source-20261010/VIX3M.csv",
            ]
        elif name == "semiconductor_volatility_target":
            required += [
                "src/us_quant/joint_risk_confirmation.py", "config/joint-risk-confirmation.json",
                "evidence/joint_risk_confirmation_20261011_registration.json",
                "src/us_quant/sector_term_guard.py", "config/sector-term-guard.json",
                "src/us_quant/sector_growth_balance.py", "config/sector-growth-balance.json",
                "src/us_quant/growth_factor_satellite.py", "config/growth-factor-satellite.json",
                "src/us_quant/multifactor_stability.py", "src/us_quant/dollar_risk_guard.py",
                "config/dollar-risk-guard.json", "src/us_quant/macro_factor_tilt.py",
                "src/us_quant/volatility_term_risk.py", "config/volatility-term-risk.json",
                "data/dollar-information-source-20261011/verified-manifest.json",
                "data/macro-rate-access-20261010/verified-manifest.json",
                "data/macro-rate-access-20261010/DFII10-curl-response.csv",
                "data/sector-term-source-20261011/verified-manifest.json",
                "data/volatility-term-source-20261010/manifest.json",
                "data/volatility-term-source-20261010/VIX.csv", "data/volatility-term-source-20261010/VIX3M.csv",
            ]
        elif name == "semiconductor_trend_transfer":
            required += [
                "src/us_quant/trend_factor_guard.py", "config/trend-factor-guard.json",
                "config/allocation-research.json",
                "evidence/dollar_risk_guard_20261011_registration.json",
                "src/us_quant/sector_growth_balance.py", "config/sector-growth-balance.json",
                "src/us_quant/growth_factor_satellite.py", "config/growth-factor-satellite.json",
                "src/us_quant/multifactor_stability.py", "src/us_quant/dollar_risk_guard.py",
                "config/dollar-risk-guard.json", "src/us_quant/macro_factor_tilt.py",
                "data/dollar-information-source-20261011/verified-manifest.json",
                "data/macro-rate-access-20261010/verified-manifest.json",
                "data/macro-rate-access-20261010/DFII10-curl-response.csv",
            ]
        elif name == "residual_treasury_rotation":
            required += [
                "src/us_quant/semiconductor_trend_transfer.py", "config/semiconductor-trend-transfer.json",
                "evidence/trend_transfer_20261011_registration.json",
                "src/us_quant/defensive_factor_rotation.py", "config/defensive-factor-rotation.json",
                "src/us_quant/trend_factor_guard.py", "config/trend-factor-guard.json",
                "config/allocation-research.json",
                "src/us_quant/factor_gold_risk.py", "config/factor-gold-risk.json",
                "src/us_quant/sector_growth_balance.py", "config/sector-growth-balance.json",
                "src/us_quant/growth_factor_satellite.py", "config/growth-factor-satellite.json",
                "src/us_quant/multifactor_stability.py", "src/us_quant/dollar_risk_guard.py",
                "config/dollar-risk-guard.json", "src/us_quant/macro_factor_tilt.py",
                "data/dollar-information-source-20261011/verified-manifest.json",
                "data/macro-rate-access-20261010/verified-manifest.json",
                "data/macro-rate-access-20261010/DFII10-curl-response.csv",
            ]
        elif name == "sector_breadth_risk":
            required += [
                "src/us_quant/sector_basket_sources.py", "config/sector-basket-sources.json",
                "src/us_quant/factor_research.py", "config/factor-research.json",
                "src/us_quant/dual_horizon.py", "config/dual-horizon.json",
                "evidence/factor_round_20261007_registration.json",
                "evidence/factor_gold_risk_20261010_registration.json",
                "evidence/sector_basket_source_readiness_20261011.json",
                "data/sector-basket-sources-20261011/manifest.json",
                "evidence/dollar_risk_guard_20261011_registration.json",
                "src/us_quant/sector_growth_balance.py", "config/sector-growth-balance.json",
                "src/us_quant/growth_factor_satellite.py", "config/growth-factor-satellite.json",
                "src/us_quant/multifactor_stability.py", "src/us_quant/dollar_risk_guard.py",
                "config/dollar-risk-guard.json", "config/trend-factor-guard.json",
                "src/us_quant/macro_factor_tilt.py",
                "data/dollar-information-source-20261011/verified-manifest.json",
                "data/macro-rate-access-20261010/verified-manifest.json",
                "data/macro-rate-access-20261010/DFII10-curl-response.csv",
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
    if name == "growth_portfolio_protection":
        from us_quant.growth_portfolio_protection import monthly_targets, run

        if key not in cache:
            cache[key] = monthly_targets(data, policy)
        return run(data, cache[key], candidate, policy, start, end, cost, delay)[1]
    if name == "semiconductor_portfolio_protection":
        from us_quant.semiconductor_portfolio_protection import monthly_baselines, run

        if key not in cache:
            cache[key] = monthly_baselines(data, policy)
        return run(data, cache[key][spec["id"]], candidate, policy, start, end, cost, delay)[1]
    if key not in cache:
        if name == "adaptive_factor_allocation":
            from us_quant.adaptive_factor_allocation import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "downside_risk_balance":
            from us_quant.downside_risk_balance import build_targets

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
        elif name == "sector_growth_balance":
            from us_quant.sector_growth_balance import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "dollar_risk_guard":
            from us_quant.dollar_risk_guard import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "financial_conditions_guard":
            from us_quant.financial_conditions_guard import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "sector_term_guard":
            from us_quant.sector_term_guard import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "inflation_factor_guard":
            from us_quant.inflation_factor_guard import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "claims_factor_guard":
            from us_quant.claims_factor_guard import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "factor_covariance_risk":
            from us_quant.factor_covariance_risk import build_targets

            cache[key] = build_targets(data, policy)[0][spec["id"]]
        elif name == "sector_basket_strategy":
            from us_quant.sector_basket_strategy import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "joint_risk_confirmation":
            from us_quant.joint_risk_confirmation import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "semiconductor_volatility_target":
            from us_quant.semiconductor_volatility_target import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "semiconductor_trend_transfer":
            from us_quant.semiconductor_trend_transfer import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "residual_treasury_rotation":
            from us_quant.residual_treasury_rotation import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
        elif name == "sector_breadth_risk":
            from us_quant.sector_breadth_risk import build_targets

            cache[key] = build_targets(data, policy)[spec["id"]]
    return seed_window(cache[key], start)
