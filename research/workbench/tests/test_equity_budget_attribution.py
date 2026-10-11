from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import equity_budget_attribution as attribution
from us_quant.config import QuantError
from us_quant.multifactor_attribution import spy_budget_ablation
from us_quant.multifactor_stability import BASE, FACTORS
from us_quant.storage import file_digest, read_json, write_json

ROOT = Path(__file__).parents[1]


def targets(columns=(*BASE, *FACTORS, "TLT", "QQQ", "SOXX")):
    dates = pd.date_range("2026-09-01", periods=4, freq="B")
    frame = pd.DataFrame(np.nan, index=dates, columns=columns)
    row = pd.Series(0.0, index=columns)
    row.loc[list(FACTORS)] = 0.1
    row["SPY"], row["GLD"], row["IEF"], row["BIL"] = 0.1, 0.25, 0.1, 0.03
    if "TLT" in columns:
        row["TLT"], row["QQQ"], row["SOXX"] = 0.05, 0.03, 0.02
    else:
        row["GLD"] += 0.1
    frame.iloc[0] = row
    frame.iloc[-1] = row
    return frame


def test_all_equity_targets_including_existing_spy_collapse_without_changing_defense_or_budget():
    original = targets()
    control = attribution.spy_budget_control(original)
    assert control.isna().equals(original.isna())
    pd.testing.assert_frame_equal(
        control[list(attribution.DEFENSE)],
        original[list(attribution.DEFENSE)],
    )
    defined = original.dropna(how="all")
    eq = [name for name in original if name in attribution.EQUITY]
    np.testing.assert_allclose(control.loc[defined.index, "SPY"], defined[eq].sum(axis=1))
    assert control.loc[defined.index, [name for name in eq if name != "SPY"]].eq(0).all().all()
    np.testing.assert_allclose(
        control.loc[defined.index].sum(axis=1),
        defined.sum(axis=1),
        rtol=0,
        atol=1e-15,
    )


def test_generalized_control_exactly_preserves_the_earlier_eight_asset_prior_art():
    original = targets((*BASE, *FACTORS))
    original.loc[original.dropna(how="all").index, "SPY"] = 0.0
    pd.testing.assert_frame_equal(
        attribution.spy_budget_control(original),
        spy_budget_ablation(original),
    )


def test_control_does_not_change_future_independent_decisions_or_fill_standby():
    original = targets()
    before = attribution.spy_budget_control(original)
    original.iloc[-1] = original.iloc[-1] * 0.5
    after = attribution.spy_budget_control(original)
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])
    assert after.iloc[1:3].isna().all().all()
    assert after.iloc[-1].sum() == pytest.approx(before.iloc[-1].sum() / 2)


@pytest.mark.parametrize(
    "problem",
    [
        "unknown_asset",
        "partial_row",
        "negative",
        "overspent",
        "duplicate_column",
        "duplicate_date",
        "missing_spy",
        "empty",
    ],
)
def test_unknown_or_invalid_allocations_are_not_safe_numeric_controls(problem):
    original = targets()
    if problem == "unknown_asset":
        original["unverified_dividend_fund"] = original["BIL"]
    elif problem == "partial_row":
        original.iloc[1, 0] = 0.2
    elif problem == "negative":
        original.iloc[0, 0] = -0.2
    elif problem == "overspent":
        original.iloc[0] *= 2
    elif problem == "duplicate_column":
        original.columns = [*original.columns[:-1], original.columns[0]]
    elif problem == "duplicate_date":
        original.index = [original.index[0]] * len(original)
    elif problem == "missing_spy":
        original = original.drop(columns="SPY")
    else:
        original[:] = np.nan
    with pytest.raises(QuantError):
        attribution.spy_budget_control(original)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_both_descriptive_legs_use_the_same_funded_rules_and_independent_accounts(
    market_factory,
    cost,
    delay,
):
    columns = (*BASE, *FACTORS, "TLT", "QQQ", "SOXX")
    data = market_factory("2020-01-02", "2020-06-30", columns)
    signals = pd.DataFrame(np.nan, index=data.close.index, columns=columns)
    row = targets().iloc[0]
    signals.iloc[0] = row
    signals.loc["2020-02-28"] = row
    signals.loc["2020-04-30"] = row * 0.8
    accounts, metadata = attribution.paired_accounts(
        data,
        signals,
        "2020-01-03",
        "2020-06-30",
        cost,
        delay,
    )
    for _, account, independent in accounts.values():
        assert account.frame.index.equals(data.close.loc["2020-01-03":].index)
        assert account.frame["cash"].min() >= 0
        np.testing.assert_allclose(
            account.frame["equity"],
            independent["equity"],
            rtol=0,
            atol=1e-8,
        )
    assert metadata["source_rule_funded_v2"]["max_independent_equity_error_usd"] < 1e-8
    difference = metadata["source_minus_budget_control"]
    for metric in ("cagr", "sharpe", "max_drawdown"):
        assert difference[metric] == pytest.approx(
            metadata["source_rule_funded_v2"]["metrics"][metric]
            - metadata["same_equity_budget_spy_control"]["metrics"][metric],
        )


def test_pair_cannot_quietly_use_another_cost_or_execution_delay(market_factory):
    data = market_factory("2020-01-02", "2020-02-28", (*BASE, *FACTORS))
    signals = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    signals.iloc[0] = targets((*BASE, *FACTORS)).iloc[0]
    with pytest.raises(QuantError, match="base or stress"):
        attribution.paired_accounts(data, signals, "2020-01-03", "2020-02-28", 0, 1)


@pytest.fixture
def policy():
    return read_json(ROOT / "config/equity-budget-attribution.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("included_candidate_count", 1),
        ("equity_symbols", ["SPY", "SOXX"]),
        ("non_equity_symbols", ["BIL", "GLD"]),
        ("paired_accounting", "silently_rewrite_original_legacy"),
        ("horizons_years", [5]),
        ("base_cost_bps", 0),
        ("stress_delay_sessions", 1),
        ("derived_control_portfolios", 0),
        ("new_strategy_evaluations", 36),
        ("independent_alpha_or_risk_matching_claimed", True),
        ("order_authority", True),
        ("investment_objective_verified", True),
    ],
)
def test_descriptive_scope_cannot_be_retuned_or_promoted(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        attribution.validate_policy(policy)


@pytest.fixture
def publication(tmp_path, monkeypatch, policy):
    root = tmp_path / "repo/research/workbench"
    root.mkdir(parents=True)
    monkeypatch.setattr(attribution, "ROOT", root)
    monkeypatch.setattr(attribution, "POLICY", root / "config/equity-budget-attribution.json")
    names = {
        "config/equity-budget-attribution.json",
        *(
            f"src/us_quant/{name}"
            for name in (
                "equity_budget_attribution.py",
                "cash_funded_accounting_v2.py",
                "backtest.py",
                "bt_audit.py",
                "factor_validation.py",
                "selection_validation.py",
                "research_weekly.py",
                "candidate_receipts.py",
                "review_receipts.py",
                "data.py",
                "calendar.py",
                "storage.py",
            )
        ),
        *(policy[key]["path"] for key in attribution.REFERENCES),
    }
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic approved frozen source")
    bundled = {"src/us_quant/backtest.py", "src/us_quant/data.py", "src/us_quant/storage.py"}
    write_json(
        root / "IMPORT_MANIFEST.json",
        {
            "file_sha256": {name: file_digest(root / name) for name in bundled},
        },
    )
    prefix = "research/workbench"
    repo = root.parents[1]
    write_json(
        repo / "config/publication_allowlist.json",
        {
            "files": [f"{prefix}/{name}" for name in sorted(names - bundled)],
            "bundle_manifest": f"{prefix}/IMPORT_MANIFEST.json",
            "bundle_manifest_sha256": file_digest(root / "IMPORT_MANIFEST.json"),
        },
    )
    calls = []

    def remote_check(root, reference, required, commit):
        assert not set(required).intersection(bundled)
        calls.append("actual_remote_verifier_called")
        return {"published_commit": commit}

    monkeypatch.setattr(attribution, "verify_publication", remote_check)
    monkeypatch.setattr(
        attribution, "git_read", lambda root, args: (repo / args[1].split(":", 1)[1]).read_bytes()
    )
    return root, repo, names, bundled, calls


def test_publication_crosschecks_approved_imported_core_without_duplicate_allowlist_entries(
    publication,
    policy,
):
    _, _, _, bundled, calls = publication
    result = attribution.published_inputs({}, [], policy, "a" * 40)
    assert calls == ["actual_remote_verifier_called"]
    assert result["verified_imported_core_file_count"] == len(bundled)
    assert result["published_import_manifest_verified"]


def test_a_published_manifest_or_core_mismatch_cannot_be_accepted(publication, policy, monkeypatch):
    monkeypatch.setattr(attribution, "git_read", lambda *args: b"not the published frozen import")
    with pytest.raises(QuantError, match="published import manifest or core"):
        attribution.published_inputs({}, [], policy, "a" * 40)


def test_unknown_unapproved_dependency_is_rejected_before_remote_claim(publication, policy):
    root, repo, _, _, calls = publication
    path = repo / "config/publication_allowlist.json"
    allowlist = read_json(path)
    allowlist["files"].remove("research/workbench/src/us_quant/equity_budget_attribution.py")
    write_json(path, allowlist)
    with pytest.raises(QuantError, match="not approved"):
        attribution.published_inputs({}, [], policy, "a" * 40)
    assert calls == []
    assert (root / "IMPORT_MANIFEST.json").exists()
