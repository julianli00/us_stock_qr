from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.research_program import ResearchProgram, engine_fingerprint, safe_file
from us_quant.selection_validation import fingerprint as account_fingerprint
from us_quant.selection_validation import original_bundle, verified_accounts
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/factor-exposure.json"
MODELS = {
    "market_full_window": ["SPY_excess"],
    "market_sector_defense_full_window": [
        "SPY_excess",
        "SOXX_minus_SPY",
        "GLD_excess",
        "IEF_excess",
    ],
    "ff6_available_overlap": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
}
REFERENCES = (
    "research_snapshot",
    "source_bound_audit",
    "selection_results",
    "source_manifest",
    "risk_proxy_bundle",
)


def read_daily_archive(path: Path, member: str, columns: list[str]) -> pd.DataFrame:
    try:
        with ZipFile(path) as archive:
            if archive.namelist() != [member]:
                raise QuantError("Use the exact single-member primary daily factor archive.")
            text = archive.read(member).decode("utf-8-sig")
    except (BadZipFile, UnicodeDecodeError) as exc:
        raise QuantError("The primary daily factor archive is not a valid CSV ZIP.") from exc
    if "created by using the 202608 CRSP database" not in text:
        raise QuantError("The archived factor data must identify its actual CRSP database cut.")
    rows = list(csv.reader(io.StringIO(text)))
    header = ["", *columns]
    positions = [i for i, row in enumerate(rows) if [cell.strip() for cell in row] == header]
    if len(positions) != 1:
        raise QuantError("The exact daily factor column header is missing or duplicated.")
    observations = []
    ended = False
    for row in rows[positions[0] + 1 :]:
        row = [cell.strip() for cell in row]
        if not row or not any(row) or row[0].startswith("Copyright"):
            ended = True
            continue
        if ended or len(row) != len(header) or len(row[0]) != 8 or not row[0].isdigit():
            raise QuantError("Unexpected sections or incomplete rows in the daily factor archive.")
        try:
            day = pd.to_datetime(row[0], format="%Y%m%d", errors="raise")
            values = [float(value) for value in row[1:]]
        except (ValueError, OverflowError) as exc:
            raise QuantError(
                "Invalid dates or numeric returns in the primary factor archive."
            ) from exc
        if not np.isfinite(values).all() or any(value in (-99.99, -999) for value in values):
            raise QuantError(
                "Missing factor sentinels cannot become returns or filled observations."
            )
        observations.append([day, *values])
    if not observations:
        raise QuantError("The primary factor archive contains no usable daily observations.")
    frame = pd.DataFrame(observations, columns=["session", *columns]).set_index("session")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise QuantError("Daily factors must have unique chronological observations.")
    return frame.astype(float) / 100


def coverage(dates: pd.DatetimeIndex, factors: pd.DataFrame) -> dict:
    end = pd.Timestamp("2026-08-31")
    if factors.index[-1] != end:
        raise QuantError(
            "The actual factor source cut changed; explicitly register a new diagnostic."
        )
    available, missing = dates[dates <= end], dates[dates > end]
    if (
        len(available) < 252
        or not factors.index.is_unique
        or not factors.index.is_monotonic_increasing
        or not available.isin(factors.index).all()
        or not np.isfinite(factors.loc[available].to_numpy(dtype=float)).all()
    ):
        raise QuantError(
            "No dropping, filling or carrying gaps within the declared factor overlap."
        )
    return {
        "formal_start": str(dates[0].date()),
        "formal_end": str(dates[-1].date()),
        "formal_sessions": len(dates),
        "available_overlap_start": str(available[0].date()),
        "available_overlap_end": str(available[-1].date()),
        "available_overlap_sessions": len(available),
        "unavailable_tail_sessions": [str(day.date()) for day in missing],
        "covers_complete_qualification_window": len(missing) == 0,
        "formal_qualification_window_changed": False,
    }


def fit_pair(factors: pd.DataFrame, excess: pd.DataFrame) -> dict:
    if (
        list(excess.columns) != ["strategy", "spy"]
        or not excess.index.equals(factors.index)
        or not factors.index.is_unique
        or not factors.index.is_monotonic_increasing
        or not factors.columns.is_unique
        or "intercept" in factors.columns
        or not all(isinstance(name, str) for name in factors.columns)
        or len(factors) < max(100, len(factors.columns) + 3)
        or factors.empty
        or not np.isfinite(factors.to_numpy(dtype=float)).all()
        or not np.isfinite(excess.to_numpy(dtype=float)).all()
    ):
        raise QuantError("Exposure fits require finite, aligned and complete paired observations.")
    x = np.column_stack([np.ones(len(factors)), factors.to_numpy(dtype=float)])
    y = excess.to_numpy(dtype=float)
    coefficients, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1]:
        raise QuantError("A rank-deficient risk model cannot identify factor exposures.")
    residual = y - x @ coefficients
    total = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    if (total <= 1e-15).any():
        raise QuantError("Constant account excess returns cannot identify explained risk.")
    names = ["intercept", *factors.columns]
    result = {
        "sessions": len(factors),
        "start": str(factors.index[0].date()),
        "end": str(factors.index[-1].date()),
        "design_condition_number": float(singular[0] / singular[-1]),
        "in_sample_descriptive_fit_only": True,
        "independent_alpha_proven": False,
    }
    for index, label in enumerate(excess.columns):
        components = coefficients[1:, index] * factors.mean().to_numpy()
        mean = float(y[:, index].mean())
        error = abs(mean - float(coefficients[0, index] + components.sum()))
        if error > 1e-12:
            raise QuantError("The arithmetic mean decomposition did not reproduce the account.")
        result[label] = {
            "coefficients": dict(zip(names, coefficients[:, index].tolist(), strict=True)),
            "annualized_252_arithmetic_intercept": float(coefficients[0, index] * 252),
            "mean_daily_excess_return": mean,
            "mean_daily_factor_components": dict(
                zip(factors.columns, components.tolist(), strict=True)
            ),
            "mean_decomposition_error": error,
            "r_squared": float(1 - (residual[:, index] ** 2).sum() / total[index]),
            "annualized_residual_volatility": float(residual[:, index].std(ddof=1) * np.sqrt(252)),
        }
    result["paired_strategy_minus_spy"] = {
        "annualized_252_arithmetic_intercept": float(
            (coefficients[0, 0] - coefficients[0, 1]) * 252
        ),
        "mean_daily_return_advantage": float((y[:, 0] - y[:, 1]).mean()),
        "coefficients": dict(
            zip(names, (coefficients[:, 0] - coefficients[:, 1]).tolist(), strict=True)
        ),
    }
    return result


def validate_policy(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "diagnostic_id": "factor_exposure_20261011",
        "as_of": "2026-10-05",
        "data_start": "2015-08-10",
        "horizons_years": [10, 5],
        "scenarios": ["base", "stress"],
        "included_candidate_count": 36,
        "total_disclosed_configurations": 120,
        "capital_usd": 10000.0,
        "factor_overlap_last_session": "2026-08-31",
        "models": MODELS,
        "highlighted_reference_ids": [
            "semiconductor_dollar_half_guard",
            "four_factor_semiconductor_sector50",
            "macro_real_yield_factor_tilt",
        ],
        "new_strategy_evaluations": 0,
    }
    if (
        any(policy.get(key) != value for key, value in expected.items())
        or any(
            policy.get(key) is not False
            for key in (
                "order_authority",
                "qualification_policy_changed",
                "forward_rules_changed",
                "fit_is_out_of_sample_forecast",
                "register_queued_factors_early",
            )
        )
        or any(
            not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"}
            for key in REFERENCES
        )
    ):
        raise QuantError(
            "The fixed descriptive models, source scope or research authority changed."
        )


def frozen_inputs(policy: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    validate_policy(policy)
    values = {
        key: read_json(safe_file(ROOT, policy[key]["path"], policy[key]["sha256"]))
        for key in REFERENCES
    }
    state, audit = values["research_snapshot"], values["source_bound_audit"]
    ids = [row["candidate_id"] for row in state["candidate_reviews"]]
    if (
        len(ids) != 36
        or len(set(ids)) != 36
        or state["pending_candidate_ids"]
        or state["total_evaluated_configurations"] != 120
        or audit["reviewed_candidates"] != 36
        or audit["regenerated_strategy_paths"] != 144
        or audit["ledger_unchanged"] is not True
        or audit["event_chain_sha256"] != state["event_chain_sha256"]
        or audit["engine_sha256"] != engine_fingerprint()
        or {row["candidate_id"] for row in audit["reviews"]} != set(ids)
    ):
        raise QuantError("Exposure attribution must retain every exact audited completed review.")
    manifest = values["source_manifest"]
    for source in manifest["sources"].values():
        safe_file(ROOT, source["path"], source["sha256"])
    five = read_daily_archive(
        ROOT / manifest["sources"]["five-factor-daily.zip"]["path"],
        "F-F_Research_Data_5_Factors_2x3_daily.csv",
        ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    )
    momentum = read_daily_archive(
        ROOT / manifest["sources"]["momentum-daily.zip"]["path"],
        "F-F_Momentum_Factor_daily.csv",
        ["Mom"],
    )
    factors = five.join(momentum, how="left")
    proxy = values["risk_proxy_bundle"]["market"]["close"]
    close = pd.read_csv(
        safe_file(ROOT, proxy["path"], proxy["sha256"]),
        index_col=0,
        parse_dates=True,
    )
    expected_dates = sessions(policy["data_start"], policy["as_of"])
    if (
        not close.index.equals(expected_dates)
        or not {"SPY", "SOXX", "GLD", "IEF"}.issubset(close.columns)
        or not np.isfinite(close[["SPY", "SOXX", "GLD", "IEF"]]).all().all()
        or (close[["SPY", "SOXX", "GLD", "IEF"]] <= 0).any().any()
    ):
        raise QuantError(
            "The frozen traded-risk proxies need the full exact positive price history."
        )
    proxy_review = next(
        row
        for row in state["candidate_reviews"]
        if row["candidate_id"] == "four_factor_semiconductor_sector50"
    )
    if digest_json(values["risk_proxy_bundle"]) != proxy_review["evidence_sha256"]:
        raise QuantError("Risk proxies must come from the exact retained sector review bundle.")
    reports = {}
    for years in policy["horizons_years"]:
        dates = sessions(
            pd.Timestamp(policy["as_of"]) - pd.DateOffset(years=years) + pd.Timedelta(days=1),
            policy["as_of"],
        )
        reports[str(years)] = coverage(dates, factors)
    return state, factors, close.pct_change(fill_method=None), reports


def fingerprint() -> str:
    return digest_json(
        {
            "exposure_module": file_digest(Path(__file__)),
            "unchanged_account_reader": account_fingerprint(),
        }
    )


def local_output(path: Path) -> Path:
    path = path if path.is_absolute() else ROOT / path
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Exposure evidence must stay in the isolated workbench.")
    return path


def register(output: Path) -> dict:
    policy = read_json(POLICY)
    state, _, _, reports = frozen_inputs(policy)
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3",
        read_json(ROOT / "config/research-program.json"),
        root=ROOT,
    )
    try:
        if program.status() != state:
            raise QuantError("The exposure study needs the exact unchanged current research state.")
        output = local_output(output)
        new_output_directory(output)
        record = {
            "schema_version": 1,
            "registered_at": utc_now(),
            "policy": policy,
            "policy_sha256": file_digest(POLICY),
            "auditor_sha256": fingerprint(),
            "program_engine_sha256": engine_fingerprint(),
            "event_chain_sha256": state["event_chain_sha256"],
            "included_candidate_ids": [row["candidate_id"] for row in state["candidate_reviews"]],
            "factor_source_coverage": reports,
            "exposure_results_computed": False,
            "new_strategy_evaluations": 0,
            "order_authority": False,
        }
        write_json(output / "registration.json", record)
        return record
    finally:
        program.close()


def run(registration_path: Path, output: Path) -> dict:
    registration_path = local_output(registration_path)
    registration = read_json(safe_file(ROOT, registration_path.relative_to(ROOT).as_posix()))
    policy = read_json(POLICY)
    state, factors, proxies, reports = frozen_inputs(policy)
    if (
        registration["policy"] != policy
        or registration["policy_sha256"] != file_digest(POLICY)
        or registration["auditor_sha256"] != fingerprint()
        or registration["program_engine_sha256"] != engine_fingerprint()
        or registration["event_chain_sha256"] != state["event_chain_sha256"]
        or registration["included_candidate_ids"]
        != [row["candidate_id"] for row in state["candidate_reviews"]]
        or registration["factor_source_coverage"] != reports
        or registration["exposure_results_computed"] is not False
        or pd.Timestamp(registration["registered_at"]).tzinfo is None
        or pd.Timestamp(registration["registered_at"]) > pd.Timestamp(utc_now())
    ):
        raise QuantError(
            "Exposure results require the actual preregistration and unchanged sources."
        )
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3",
        read_json(ROOT / "config/research-program.json"),
        root=ROOT,
    )
    try:
        if program.status() != state:
            raise QuantError("The research state changed after exposure preregistration.")
        studies, sources = {}, {}
        for review in state["candidate_reviews"]:
            name = review["candidate_id"]
            bundle, sources[name] = original_bundle(review)
            source = bundle["market"]["risk_free"]
            rates = pd.read_csv(
                safe_file(ROOT, source["path"], source["sha256"]),
                index_col=0,
                parse_dates=True,
            )
            studies[name] = {}
            for years in policy["horizons_years"]:
                accounts = verified_accounts(bundle, review, years, policy)
                for scenario in policy["scenarios"]:
                    own, spy = accounts[scenario]["strategy"], accounts[scenario]["spy"]
                    dates = own.index
                    rf = rates.loc[dates, "risk_free"]
                    paired = pd.DataFrame({"strategy": own["return"], "spy": spy["return"]})
                    x = pd.DataFrame(
                        {
                            "SPY_excess": proxies.loc[dates, "SPY"] - rf,
                            "SOXX_minus_SPY": proxies.loc[dates, "SOXX"]
                            - proxies.loc[dates, "SPY"],
                            "GLD_excess": proxies.loc[dates, "GLD"] - rf,
                            "IEF_excess": proxies.loc[dates, "IEF"] - rf,
                        }
                    )
                    overlap = dates[dates <= pd.Timestamp(policy["factor_overlap_last_session"])]
                    ff = factors.loc[overlap]
                    models = {
                        "market_full_window": fit_pair(
                            x[MODELS["market_full_window"]], paired.sub(rf, axis=0)
                        ),
                        "market_sector_defense_full_window": fit_pair(x, paired.sub(rf, axis=0)),
                        "ff6_available_overlap": fit_pair(
                            ff[MODELS["ff6_available_overlap"]],
                            paired.loc[overlap].sub(ff["RF"], axis=0),
                        ),
                    }
                    studies[name][f"{years}y_{scenario}"] = {
                        "models": models,
                        "ff_risk_free_minus_frozen_account_rf_daily_mean": float(
                            (ff["RF"] - rf.loc[overlap]).mean()
                        ),
                        "qualification_metrics_recomputed_or_changed": False,
                    }
        if program.status() != state:
            raise QuantError("Read-only exposure attribution cannot change or race research state.")
        result = {
            "schema_version": 1,
            "created_at": utc_now(),
            "diagnostic_id": policy["diagnostic_id"],
            "registration": {
                "path": registration_path.relative_to(ROOT).as_posix(),
                "sha256": file_digest(registration_path),
            },
            "auditor_sha256": fingerprint(),
            "program_engine_sha256": engine_fingerprint(),
            "event_chain_sha256": state["event_chain_sha256"],
            "included_candidate_count": 36,
            "original_reviewed_paths": 144,
            "paired_regressions": 432,
            "factor_source_coverage": reports,
            "studies": studies,
            "source_bundles": sources,
            "ledger_unchanged": True,
            "qualification_policy_changed": False,
            "descriptive_arithmetic_intercepts_are_not_CAGR_or_tradable_alpha": True,
            "new_strategy_evaluations": 0,
            "independent_alpha_proven": False,
            "investment_objective_verified": False,
            "order_authority": False,
        }
        output = local_output(output)
        new_output_directory(output)
        write_json(output / "results.json", result)
        return result
    finally:
        program.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only factor exposure attribution; never orders."
    )
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument(
        "--registration",
        type=Path,
        default=ROOT / "data/factor-exposure-20261011/registration.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / (
        "data/factor-exposure-20261011"
        if args.action == "register"
        else "reports/factor-exposure-20261011/primary"
    )
    try:
        result = register(output) if args.action == "register" else run(args.registration, output)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key not in ("policy", "studies", "source_bundles", "included_candidate_ids")
                },
                indent=2,
            )
        )
    except QuantError as exc:
        parser.exit(2, f"Factor exposure diagnostic blocked: {exc}\n")


if __name__ == "__main__":
    main()
