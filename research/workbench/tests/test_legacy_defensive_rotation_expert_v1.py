from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import legacy_defensive_rotation_expert_v1 as study
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


def test_actual_identity_does_not_require_financial_pass():
    policy = read_json(study.POLICY)
    snapshot = read_json(study.base.checked_reference(policy["source_identity_snapshot"]))
    source = study.check_identity(
        policy,
        snapshot,
        read_json(study.base.checked_reference(policy["source_registration"])),
        read_json(study.base.checked_reference(policy["source_results"])),
        read_json(study.base.checked_reference(policy["source_bundle"])),
    )
    assert source["id"] == study.SOURCE
    assert snapshot["review"]["historical_gates_passed"] is False
    assert snapshot["review"]["status"] == "rejected_historical"


def test_immutable_original_identity_has_no_technical_failure():
    policy = read_json(study.POLICY)
    record = study.legacy_snapshot(policy)
    assert record["spec_sha256"] == policy["source_specification_sha256"]
    assert record["sqlite_mode"] == "ro_immutable_query_only"


@pytest.mark.parametrize("damage", ["spec_hash", "duplicate_registration", "accounting_contract"])
def test_changed_source_identity_is_not_an_admissible_expert(damage):
    policy = read_json(study.POLICY)
    snapshot = read_json(study.base.checked_reference(policy["source_identity_snapshot"]))
    packet = read_json(study.base.checked_reference(policy["source_registration"]))
    results = read_json(study.base.checked_reference(policy["source_results"]))
    bundle = read_json(study.base.checked_reference(policy["source_bundle"]))
    if damage == "spec_hash":
        policy["source_specification_sha256"] = "0" * 64
    elif damage == "duplicate_registration":
        packet["candidates"].append(
            copy.deepcopy(next(r for r in packet["candidates"] if r["spec"]["id"] == study.SOURCE))
        )
    else:
        bundle["accounting_engine"] = {"name": "mislabelled_v2"}
    with pytest.raises(QuantError):
        study.check_identity(policy, snapshot, packet, results, bundle)


def lineage_frames():
    dates = sessions("2020-01-02", "2020-01-06")
    native = {
        k: pd.DataFrame(0.1, index=dates, columns=["SPY", "BIL"])
        for k in ("open", "close", "raw_close", "volume")
    }
    native["risk_free"] = pd.DataFrame(0.00001, index=dates, columns=["risk_free"])
    stored = {k: v.copy() for k, v in native.items()}
    active = {k: v.copy() for k, v in native.items()}
    for k in ("open", "close", "raw_close", "risk_free"):
        active[k].iloc[0, 0] = np.nextafter(active[k].iloc[0, 0], np.inf)
    return native, stored, active


def test_exact_serialization_lineage_is_not_a_tolerance():
    native, stored, active = lineage_frames()
    result = study.check_lineage(native, stored, active)
    assert all(r["source_native_equals_common_stored_roundtrip_exactly"] for r in result.values())
    assert result["risk_free"]["source_vs_active_common_native_exact"] is False
    assert result["risk_free"]["source_vs_active_common_native_different_cells"] == 1
    stored["risk_free"].iloc[1, 0] = np.nextafter(stored["risk_free"].iloc[1, 0], np.inf)
    with pytest.raises(QuantError, match="exact stored"):
        study.check_lineage(native, stored, active)


def target_fixture():
    calendar = sessions("2020-01-02", "2020-03-02")
    targets = pd.DataFrame(np.nan, index=calendar, columns=study.ASSETS)
    first = pd.Series(0.0, index=study.ASSETS)
    first.loc[list(study.FACTORS)] = 0.49 / 4
    first["GLD"] = 0.49
    second = first.copy()
    second.loc[list(study.FACTORS)] = 0.294 / 4
    second[["GLD", "BIL"]] = [0.0, 0.686]
    targets.loc[["2020-01-31", "2020-02-03"]] = first.to_numpy()
    targets.loc["2020-02-28"] = second
    return targets


def test_only_duplicate_initial_seed_is_removed():
    original = target_fixture()
    derived = study.nominal_view(original, "2020-02-03")
    assert derived.loc["2020-02-03"].isna().all()
    assert original.loc["2020-02-03"].notna().all()
    assert derived.dropna().index.tolist() == [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-28"),
    ]
    pd.testing.assert_frame_equal(
        derived.loc[derived.dropna().index], original.loc[derived.dropna().index], check_exact=True
    )


@pytest.mark.parametrize("damage", ["different_seed", "extra_seed", "missing_month", "partial_row"])
def test_missing_or_changed_source_intention_is_not_a_seed_default(damage):
    target = target_fixture()
    if damage == "different_seed":
        target.loc["2020-02-03", ["GLD", "BIL"]] = [0.48, 0.01]
    elif damage == "extra_seed":
        target.loc["2020-02-04"] = target.loc["2020-02-03"]
    elif damage == "missing_month":
        target.loc["2020-02-28"] = np.nan
    else:
        target.loc["2020-02-03", "GLD"] = np.nan
    with pytest.raises(QuantError):
        study.nominal_view(target, "2020-02-03")


def shadow_fixture(dates):
    returns = pd.Series(0.0003 + 0.001 * np.sin(np.arange(len(dates))), index=dates)
    nav = 10000 * (1 + returns).cumprod()
    frame = pd.DataFrame(
        {
            "equity": nav,
            "return": returns,
            "cash": nav * 0.02,
            "gross_exposure": 0.98,
            "turnover": 0.0,
            "cost": 0.0,
            "orders": 0,
            "risk_free": 0.00001,
        },
        index=dates,
    )
    return frame, frame["risk_free"].copy()


def test_assessment_RF_view_changes_only_the_declared_reference():
    dates = sessions("2016-10-06", "2016-10-12")
    native, source_rf = shadow_fixture(dates)
    common = source_rf.copy()
    common.iloc[:] = np.nextafter(common.iloc[0], np.inf)
    derived = study.assessment_view(native, source_rf, common)
    assert derived["risk_free"].equals(common)
    assert native["risk_free"].equals(source_rf)
    assert not derived["risk_free"].equals(source_rf)
    unchanged = [c for c in native if c != "risk_free"]
    pd.testing.assert_frame_equal(derived[unchanged], native[unchanged], check_exact=True)
    with pytest.raises(QuantError, match="complete native"):
        study.assessment_view(native, source_rf, common.drop(dates[1]))


def monthly_fixture():
    calendar = sessions("2016-09-01", "2017-04-28")
    history = calendar[calendar >= "2016-10-06"]
    targets = pd.DataFrame(np.nan, index=calendar, columns=study.ASSETS)
    row = pd.Series(0.0, index=study.ASSETS)
    row.loc[list(study.FACTORS)] = 0.49 / 4
    row["BIL"] = 0.49
    targets.loc[[d for d in calendar if is_month_end(d)]] = row.to_numpy()
    targets.loc["2016-10-05"] = row
    native, source_rf = shadow_fixture(history)
    common = source_rf.copy()
    common.iloc[:] = np.nextafter(common.iloc[0], np.inf)
    target_view = study.nominal_view(targets, "2016-10-05")
    rf_view = study.assessment_view(native, source_rf, common)
    universe = (*study.ASSETS, "QQQ", "XLK", "SOXX", "IJR", "PKW")
    identity = ExpertIdentity(study.SOURCE, study.GROUP, "2026-10-30T03:00:00Z")
    return identity, target_view, rf_view, universe, common


def test_existing_monthly_interface_has_known_target_but_unmatured_initial_score():
    identity, target_view, rf_view, universe, common = monthly_fixture()
    expert = ArchivedMonthlyExpert(identity, target_view, rf_view, universe)
    initial = expert.observe(pd.Timestamp("2016-10-05"), 126)
    assert initial.target_known_at == pd.Timestamp("2016-09-30")
    assert len(initial.net_returns) == 0 and initial.score_mature is False
    assert initial.target.loc[["SOXX", "IJR", "PKW"]].sum() == 0
    final = expert.observe(rf_view.index[-1], 126)
    assert final.score_mature is True and final.risk_free.equals(
        common.loc[final.net_returns.index]
    )
    assert initial.historical_counterfactual is True


def test_future_targets_and_RF_cannot_change_observed_prefix():
    identity, targets, shadow, universe, _ = monthly_fixture()
    before = pd.Timestamp("2017-01-31")
    original = ArchivedMonthlyExpert(identity, targets, shadow, universe).observe(before, 42)
    future_targets, future_shadow = targets.copy(), shadow.copy()
    future = future_targets.index[
        (future_targets.index > before) & future_targets.notna().all(axis=1)
    ]
    future_targets.loc[future] = 0.0
    future_targets.loc[future, "BIL"] = 0.98
    future_shadow.loc[future_shadow.index > before, "risk_free"] = 0.01
    changed = ArchivedMonthlyExpert(identity, future_targets, future_shadow, universe).observe(
        before, 42
    )
    assert original.target_known_at == changed.target_known_at
    pd.testing.assert_series_equal(original.target, changed.target, check_exact=True)
    pd.testing.assert_series_equal(original.net_returns, changed.net_returns, check_exact=True)
    pd.testing.assert_series_equal(original.risk_free, changed.risk_free, check_exact=True)


@pytest.mark.parametrize("damage", ["cash", "gross", "nan_cash", "BT", "missing_identity"])
def test_invalid_cached_cash_or_leverage_is_not_laundered(damage):
    policy = read_json(study.POLICY)
    proof = copy.deepcopy(
        read_json(study.base.checked_reference(policy["cached_native_funding_and_seed_proof"]))
    )
    bundle = read_json(study.base.checked_reference(policy["source_bundle"]))
    study.check_cached_funding(proof, policy["source_bundle"], bundle)
    row = proof["rows"][0]
    if damage == "cash":
        row["minimum_cash_usd"] = -0.01
    elif damage == "gross":
        row["maximum_gross_exposure"] = 1.01
    elif damage == "nan_cash":
        row["minimum_cash_usd"] = np.nan
    elif damage == "BT":
        row["cached_BT_maximum_NAV_difference_usd"] = 1
    else:
        row["cached_accounting_errors"] = {}
    with pytest.raises(QuantError):
        study.check_cached_funding(proof, policy["source_bundle"], bundle)


def publication_fixture(monkeypatch, tmp_path):
    manifest = {"file_sha256": {}}
    bundled_blob = b"frozen imported dependency\n"
    bundled_digest = hashlib.sha256(bundled_blob).hexdigest()
    manifest["file_sha256"]["src/us_quant/imported.py"] = bundled_digest
    raw = json.dumps(manifest, sort_keys=True).encode()
    manifest_path = tmp_path / "IMPORT_MANIFEST.json"
    manifest_path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    explicit = "src/us_quant/explicit.py"
    refs = [
        {"path": explicit, "sha256": "a" * 64},
        {"path": "src/us_quant/imported.py", "sha256": bundled_digest},
    ]
    calls = []
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "MANIFEST_SHA", sha)
    monkeypatch.setattr(
        study,
        "read_json",
        lambda p: (
            {"files": ["research/workbench/" + explicit], "bundle_manifest_sha256": sha}
            if p.name == "publication_allowlist.json"
            else manifest
        ),
    )
    monkeypatch.setattr(study.base, "checked_reference", lambda r: calls.append(r["path"]))
    monkeypatch.setattr(study.base, "reference", lambda p: {"path": p.name, "sha256": sha})

    def verified(root, packet, required, commit):
        assert root == tmp_path and required == {explicit} and commit == "b" * 40
        calls.append("all_explicit_checked")
        return {"published_commit": commit}

    def git(command, **kwargs):
        path = command[-1].split(":", 1)[1]
        calls.append(path)
        if path == "research/workbench/IMPORT_MANIFEST.json":
            return SimpleNamespace(stdout=raw)
        assert path == "research/workbench/src/us_quant/imported.py"
        return SimpleNamespace(stdout=bundled_blob)

    monkeypatch.setattr(study, "verify_publication", verified)
    monkeypatch.setattr(study.subprocess, "run", git)
    return refs, calls


def test_publication_checks_every_explicit_and_pinned_dependency(monkeypatch, tmp_path):
    refs, calls = publication_fixture(monkeypatch, tmp_path)
    result = study.published_source_dependencies(refs, {"path": "evidence/source.json"}, "b" * 40)
    assert result["pinned_import_dependencies_checked"] == 1
    assert all(r["path"] in calls for r in refs)
    assert "all_explicit_checked" in calls
    assert "research/workbench/IMPORT_MANIFEST.json" in calls
    assert "research/workbench/src/us_quant/imported.py" in calls


def test_unlisted_dependency_is_not_dropped_or_silently_approved(monkeypatch, tmp_path):
    refs, calls = publication_fixture(monkeypatch, tmp_path)
    refs.append({"path": "src/us_quant/unknown.py", "sha256": "c" * 64})
    with pytest.raises(QuantError, match="neither explicit nor pinned"):
        study.published_source_dependencies(refs, {"path": "evidence/source.json"}, "b" * 40)
    assert "all_explicit_checked" not in calls


def test_wrong_published_import_blob_is_rejected(monkeypatch, tmp_path):
    refs, _ = publication_fixture(monkeypatch, tmp_path)
    original = study.subprocess.run

    def wrong_blob(command, **kwargs):
        result = original(command, **kwargs)
        if command[-1].endswith("/src/us_quant/imported.py"):
            return SimpleNamespace(stdout=result.stdout + b"changed")
        return result

    monkeypatch.setattr(study.subprocess, "run", wrong_blob)
    with pytest.raises(QuantError, match="Git blob differs"):
        study.published_source_dependencies(refs, {"path": "evidence/source.json"}, "b" * 40)
