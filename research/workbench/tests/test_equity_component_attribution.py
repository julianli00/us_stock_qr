from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import equity_component_attribution as attribution
from us_quant.config import QuantError
from us_quant.storage import file_digest, read_json, write_json

ROOT = Path(__file__).parents[1]


def targets():
    columns = (*attribution.budget.EQUITY, *attribution.budget.DEFENSE)
    dates = pd.date_range("2026-09-01", periods=4, freq="B")
    result = pd.DataFrame(np.nan, index=dates, columns=columns)
    row = pd.Series(0.02, index=columns)
    row["GLD"], row["BIL"], row["IEF"], row["TLT"] = 0.4, 0.03, 0.1, 0.05
    result.iloc[0] = row
    result.iloc[-1] = row * 0.8
    return result


@pytest.mark.parametrize("component", list(attribution.COMPONENTS))
def test_component_projection_retains_unselected_targets_seed_budget_and_pauses(component):
    original = targets()
    replacement, exposed = attribution.component_control(original, component)
    assert exposed
    selected = list(attribution.COMPONENTS[component])
    active = original.dropna(how="all")
    unchanged = [name for name in original if name not in {*selected, "SPY"}]
    pd.testing.assert_frame_equal(replacement[unchanged], original[unchanged])
    assert replacement.isna().equals(original.isna())
    assert replacement.loc[active.index, selected].eq(0).all().all()
    np.testing.assert_allclose(
        replacement.loc[active.index, "SPY"],
        active["SPY"] + active[selected].sum(axis=1), rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
        replacement.loc[active.index].sum(axis=1), active.sum(axis=1), rtol=0, atol=1e-15,
    )


def test_aggregate_styles_and_satellites_cover_the_prior_whole_equity_projection():
    original = targets()
    styles, _ = attribution.component_control(original, "all_styles")
    both, _ = attribution.component_control(styles, "all_satellites")
    np.testing.assert_allclose(
        both, attribution.budget.spy_budget_control(original),
        rtol=0, atol=1e-15, equal_nan=True,
    )


def test_zero_exposure_is_an_exact_identity_not_a_new_control():
    original = targets()
    original.loc[original.dropna(how="all").index, "SOXX"] = 0
    replacement, exposed = attribution.component_control(original, "semiconductor")
    assert exposed is False
    pd.testing.assert_frame_equal(replacement, original)


def test_absent_component_assets_are_disclosed_as_identity():
    original = targets().drop(columns=["SOXX"])
    replacement, exposed = attribution.component_control(original, "semiconductor")
    assert exposed is False
    pd.testing.assert_frame_equal(replacement, original)


def test_unregistered_component_is_rejected_instead_of_selecting_a_favorable_subset():
    with pytest.raises(QuantError, match="frozen component"):
        attribution.component_control(targets(), "SOXX_only_after_2020")


def test_future_targets_cannot_change_earlier_replacements():
    original = targets()
    before, _ = attribution.component_control(original, "all_styles")
    original.iloc[-1] *= 0.5
    after, _ = attribution.component_control(original, "all_styles")
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])
    assert after.iloc[1:3].isna().all().all()


@pytest.mark.parametrize(
    "problem", ["unknown_asset", "partial", "negative", "overspent", "duplicate", "empty"],
)
def test_invalid_original_targets_explicitly_block_component_controls(problem):
    original = targets()
    if problem == "unknown_asset":
        original["NOBL_unadmitted"] = original["BIL"]
    elif problem == "partial":
        original.iloc[1, 0] = 0.1
    elif problem == "negative":
        original.iloc[0, 0] = -0.1
    elif problem == "overspent":
        original.iloc[0] *= 2
    elif problem == "duplicate":
        original.index = [original.index[0]] * len(original)
    else:
        original[:] = np.nan
    with pytest.raises(QuantError):
        attribution.component_control(original, "all_styles")


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_nonidentity_accounts_keep_funded_execution_and_independent_bt(market_factory, cost, delay):
    original = targets()
    data = market_factory("2020-01-02", "2020-06-30", tuple(original.columns))
    signals = pd.DataFrame(np.nan, index=data.close.index, columns=original.columns)
    signals.iloc[0] = original.iloc[0]
    signals.loc["2020-02-28"] = original.iloc[-1]
    replacement, _ = attribution.component_control(signals, "all_styles")
    account, independent, metadata = attribution.control_account(
        data, replacement, "2020-01-03", "2020-06-30", cost, delay,
    )
    np.testing.assert_allclose(
        account.frame["equity"], independent["equity"], rtol=0, atol=1e-8,
    )
    assert account.frame["cash"].min() >= 0
    assert metadata["metrics"]["sessions"] == len(data.close.loc["2020-01-03":])
    assert metadata["total_orders"] == account.frame["orders"].sum()


def test_accounts_reject_another_cost_or_delay(market_factory):
    data = market_factory("2020-01-02", "2020-02-28", tuple(targets().columns))
    with pytest.raises(QuantError, match="original base or stress"):
        attribution.control_account(data, targets(), "2020-01-03", "2020-02-28", 0, 1)


@pytest.fixture
def policy():
    return read_json(ROOT / "config/equity-component-attribution.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("included_candidate_count", 1),
        ("components", {"semiconductor": ["SOXX"]}),
        ("horizons_years", [5]),
        ("stress_cost_bps", 5.0),
        ("stress_delay_sessions", 1),
        ("logical_component_controls", 1),
        ("logical_horizon_scenario_comparisons", 4),
        ("new_strategy_evaluations", 288),
        ("independent_alpha_or_additive_attribution_claimed", True),
        ("existing_forward_rules_changed", True),
        ("order_authority", True),
        ("investment_objective_verified", True),
    ],
)
def test_scope_and_qualification_cannot_change(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        attribution.validate_policy(policy)


def test_summary_separates_identities_from_exposed_win_denominators():
    rows = []
    for years in (10, 5):
        for scenario in ("base", "stress"):
            for name in attribution.COMPONENTS:
                active = name == "all_styles"
                rows.append({
                    "candidate_id": "synthetic",
                    "component_id": name,
                    "years": years, "scenario": scenario,
                    "selected_component_exposure_present": active,
                    "source_minus_component_control": {
                        "cagr": 0.01 if active else 0,
                        "sharpe": None if active else 0,
                    },
                })
    result = attribution.summaries(rows)
    assert result["all_styles"]["source_cagr_above_control_nonidentity_pairs"] == 4
    assert result["all_styles"]["source_cagr_above_control_in_all_four_exposed_paths"] == 1
    assert result["semiconductor"]["identity_comparisons"] == 4
    assert result["semiconductor"]["source_cagr_above_control_nonidentity_pairs"] == 0
    assert result["all_styles"]["source_sharpe_above_control_nonidentity_pairs"] == 0


@pytest.fixture
def registration_environment(tmp_path, monkeypatch, policy):
    root = tmp_path / "research/workbench"
    root.mkdir(parents=True)
    path = root / "config/equity-component-attribution.json"
    write_json(path, policy)
    monkeypatch.setattr(attribution, "ROOT", root)
    monkeypatch.setattr(attribution, "POLICY", path)
    state = {"event_chain_sha256": "synthetic_head"}
    entries = [{"spec": {"id": f"synthetic_{number}"}, "bundle_reference": {}} for number in range(36)]
    source = {"source_results": {
        "completed_at": "2026-10-11T09:50:31+00:00",
        "private_result_source": "reports/synthetic-old/results.json",
        "private_result_source_sha256": "synthetic_sha",
    }}
    monkeypatch.setattr(attribution, "frozen_sources", lambda policy: (state, entries, source))
    monkeypatch.setattr(attribution, "utc_now", lambda: "2026-10-11T10:10:00+00:00")
    monkeypatch.setattr(
        attribution.budget, "program",
        lambda: SimpleNamespace(status=lambda: state, close=lambda: None),
    )
    return root, state, entries, source


def test_registration_is_metadata_only_and_no_overwrite(registration_environment):
    root, _, _, _ = registration_environment
    path = root / "evidence/registration.json"
    actual = attribution.register(path)
    assert actual["new_component_accounting_outcomes_computed"] is False
    assert len(actual["included_candidate_ids"]) == 36
    assert not (root / "reports").exists()
    before = file_digest(path)
    with pytest.raises(QuantError):
        attribution.register(path)
    assert file_digest(path) == before


def test_component_registration_cannot_claim_to_precede_exposed_source(
    registration_environment, monkeypatch,
):
    root, _, _, _ = registration_environment
    monkeypatch.setattr(attribution, "utc_now", lambda: "2026-10-11T09:40:00+00:00")
    with pytest.raises(QuantError, match="cannot predate"):
        attribution.register(root / "evidence/registration.json")
    assert not (root / "evidence").exists()


def test_unpublished_registration_blocks_before_accounts_or_output(
    registration_environment, monkeypatch,
):
    root, _, _, _ = registration_environment
    path = root / "evidence/registration.json"
    attribution.register(path)

    def unpublished(*args):
        raise QuantError("actual registration is not published")

    monkeypatch.setattr(attribution, "published_inputs", unpublished)
    monkeypatch.setattr(
        attribution, "control_account", lambda *args: pytest.fail("no account before publication"),
    )
    with pytest.raises(QuantError, match="not published"):
        attribution.run(path, root / "reports/new", "a" * 40)
    assert not (root / "reports").exists()


def test_tampered_registration_blocks_before_remote_or_accounts(
    registration_environment, monkeypatch,
):
    root, _, _, _ = registration_environment
    path = root / "evidence/registration.json"
    actual = attribution.register(path)
    actual["component_ids"] = ["semiconductor"]
    write_json(path, actual)
    monkeypatch.setattr(
        attribution, "published_inputs", lambda *args: pytest.fail("tampered receipt"),
    )
    with pytest.raises(QuantError, match="exact registered"):
        attribution.run(path, root / "reports/new", "a" * 40)


def test_new_publication_guard_inherits_old_bindings_and_checks_new_code_sources(
    registration_environment, policy, monkeypatch,
):
    root, _, entries, _ = registration_environment
    old = root / policy["source_budget_policy"]["path"]
    write_json(old, {})
    policy["source_budget_policy"]["sha256"] = file_digest(old)
    calls = []

    def inherited(reference, selected, original, commit):
        assert selected == entries and commit == "a" * 40
        calls.append("inherited")
        return {"published_commit": commit}

    def own(actual_root, reference, required, commit):
        assert actual_root == root
        assert "config/equity-component-attribution.json" in required
        assert "evidence/equity_budget_attribution_20261011_results.json" in required
        assert any(name.endswith("equity_component_attribution.py") for name in required)
        calls.append("own")
        return {"published_commit": commit}

    monkeypatch.setattr(attribution.budget, "published_inputs", inherited)
    monkeypatch.setattr(attribution, "verify_publication", own)
    result = attribution.published_inputs({}, entries, policy, "a" * 40)
    assert calls == ["inherited", "own"]
    assert result["all_old_and_new_frozen_dependencies_verified"]
