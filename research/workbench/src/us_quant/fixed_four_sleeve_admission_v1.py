from __future__ import annotations

import numpy as np
import pandas as pd

from us_quant import ssd_fixed_four_sleeve_financial_v1 as source
from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.config import QuantError
from us_quant.monthly_style_reversal_admission_v1 import _artifact, _frame
from us_quant.storage import digest_json, read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity, StrategyExpert

base, law = source.base, source.law
POLICY = source.ROOT / "config/fixed-four-sleeve-admission-v1.json"
FOLDER = base.REPORTS / "fixed-four-sleeve-admission-20261105-v1"
REVIEW_FREEZE = FOLDER / "review-implementation-freeze-v1.json"
ADMISSION = FOLDER / "technical-admission-v1.json"
PUBLIC = source.ROOT / "evidence/fixed_four_sleeve_financial_20261105_admission_v1.json"
MECHANISM = "fixed_cross_asset_capital_rebalancing"
DEFENSIVE = ("TLT", "GLD", "BIL")
OWN = (
    "config/fixed-four-sleeve-admission-v1.json",
    "src/us_quant/fixed_four_sleeve_admission_v1.py",
    "tests/test_fixed_four_sleeve_admission_v1.py",
    "FIXED_FOUR_SLEEVE_ADMISSION_V1.md",
)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_fixed_four_sleeve_technical_admission_20261105_v1",
        "candidate_id": law.ID,
        "mechanism_group": MECHANISM,
        "source_packet_schema": 3,
        "universe": list(law.UNIVERSE),
        "minimum_actual_style_families_per_path": 3,
        "required_actual_fixed_defensive_buckets": list(DEFENSIVE),
        "source_price_warmup_sessions": 0,
        "existing_positive_weight_threshold": 1e-10,
        "existing_independent_BT_tolerance_usd": 1e-6,
        "maximum_gross_exposure": 1.0,
        "minimum_cash_usd": 0.0,
        "score_lookback": 126,
        "RF_comparison": "exact_original_canonical_review_market_series",
        "financial_performance_is_technical_admission_gate": False,
        "new_source_requests": 0,
        "source_targets_or_accounts_replayed": False,
        "global_pool_promotion": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if (
        not isinstance(policy, dict)
        or policy != expected
        or any(type(policy[key]) is not type(value) for key, value in expected.items())
    ):
        raise QuantError(
            "Preserve original technical standards and the fixed policy's asset scope."
        )


def inspect_source(policy: dict) -> tuple[StrategyExpert, dict]:
    validate(policy)
    data, frozen, _ = source.inputs()
    packet = source.checked_packet(frozen)
    result = read_json(source.RESULTS)
    registered, reviewed, proved = [
        source.events(kind) for kind in ("registered", "review", "target_proof")
    ]
    published = source.publication_fence()
    if (
        len(registered) != 1
        or len(reviewed) != 1
        or len(proved) != 1
        or result["frozen"] != frozen
        or reviewed[0]["body"] != result["review"]
        or registered[0]["body"] != {"candidate": law.CANDIDATE, "frozen": frozen}
        or packet["entries"][0]["registered_at"] != registered[0]["at"]
        or packet["entries"][0]["registered_event_sha256"] != registered[0]["sha256"]
        or published["body"]["verification"] != result["publication"]
        or result["publication"]["registration_packet"] != source.reference(source.PACKET)
        or proved[0]["body"]["native_source_proof"] != result["native_source_proof"]
        or result["review"]["status"] not in {"rejected", "historical_point_gates_passed"}
        or result["review"]["all_four_paths_same_candidate_and_frozen_parameters"] is not True
        or result["review"]["order_authority"] is not False
        or frozen["order_authority"] is not False
        or frozen["decision_information_clock"] != law.CLOCK
        or not registered[0]["sequence"]
        < published["sequence"]
        < proved[0]["sequence"]
        < reviewed[0]["sequence"]
    ):
        raise QuantError("Require one unchanged registered and published complete source identity.")
    native = result["native_source_proof"]
    proof = read_json(base.checked_reference(native["proof"]))
    started = read_json(base.checked_reference(proof["source_start"]))
    if (
        proof["frozen_specification_sha256"] != digest_json(frozen)
        or proof["publication_event_sha256"] != published["sha256"]
        or started["publication_event_sha256"] != published["sha256"]
        or proof["real_native_targets_first_generated_after_actual_publication"] is not True
        or proof["all_independent_fractional_four_bucket_targets_exact"] is not True
        or proof["market_prices_used_for_weight_choice"] is not False
        or type(proof["source_price_binary64_inputs_checked"]) is not int
        or proof["source_price_binary64_inputs_checked"] != 0
        or proof["whole_native_prehistory_retained"] is not True
        or proof["synthetic_initial_recommendation_added"] is not False
        or not pd.Timestamp(published["at"])
        <= pd.Timestamp(started["started_at"])
        <= pd.Timestamp(proof["checked_at"])
        < pd.Timestamp(reviewed[0]["at"])
    ):
        raise QuantError("Fixed opinions must follow actual publication without price forecasts.")
    if set(result["review"]["paths"]) != base.PATH_KEYS:
        raise QuantError("Retain all four original paths before technical admission.")
    starts, completes = [source.events(kind) for kind in ("path_started", "path_completed")]
    if (
        len(starts) != 4
        or len(completes) != 4
        or {row["body"]["path_key"] for row in starts} != base.PATH_KEYS
        or {row["body"]["path_key"] for row in completes} != base.PATH_KEYS
    ):
        raise QuantError("Require one actual start and completion for every original source path.")
    paths, families, defensive = {}, {}, {}
    protocol = read_json(base.PROTOCOL)
    allowed = (*law.CORE, *DEFENSIVE)
    for window in protocol["windows"]:
        dates = sessions(window["start"], window["end"])
        for scenario in ("base", "stress"):
            key = f"{window['years']}y-{scenario}"
            reference = result["review"]["paths"][key]
            receipt = base.verify_path(reference, frozen, law.ID, key)
            begin = next(row for row in starts if row["body"]["path_key"] == key)
            end = next(row for row in completes if row["body"]["path_key"] == key)
            if (
                receipt != result["paths"][key]
                or end["body"]["receipt"] != reference
                or not published["sequence"]
                < begin["sequence"]
                < end["sequence"]
                < proved[0]["sequence"]
                or not pd.Timestamp(proof["checked_at"])
                <= pd.Timestamp(begin["at"])
                <= pd.Timestamp(end["at"])
                or receipt["execution"]["minimum_cash_usd"] < 0
                or not 0 <= receipt["execution"]["maximum_gross_exposure"] <= 1
                or receipt["independent_bt_max_equity_difference_usd"] > 1e-6
                or receipt["broker_orders"] != 0
            ):
                raise QuantError(
                    "Invalid funding or accounting cannot become an admissible expert."
                )
            account, weights = [
                _frame(_artifact(receipt, name)) for name in ("account.csv", "weights.csv")
            ]
            required = ("equity", "return", "risk_free", "cash", "gross_exposure", "cost", "orders")
            if (
                not account.index.equals(dates)
                or not weights.index.equals(dates)
                or tuple(weights.columns) != law.UNIVERSE
                or not set(required).issubset(account.columns)
                or not np.isfinite(account.loc[:, list(required)].to_numpy()).all()
                or (account["equity"] <= 0).any()
                or (account["return"] <= -1).any()
                or not np.isfinite(weights.to_numpy()).all()
                or (weights < 0).any().any()
                or (weights.sum(axis=1) > 1).any()
                or (account["cash"] < 0).any()
                or (account["gross_exposure"] < 0).any()
                or (account["gross_exposure"] > 1).any()
                or (account["cost"] < 0).any()
                or (account["orders"] < 0).any()
                or not account["risk_free"].equals(data.risk_free.loc[dates])
            ):
                raise QuantError("Do not crop source dates, borrow, remap assets or replace RF.")
            used = [symbol for symbol in law.CORE if (weights[symbol] > 1e-10).any()]
            buckets = [symbol for symbol in DEFENSIVE if (weights[symbol] > 1e-10).any()]
            if (
                len(used) < 3
                or tuple(buckets) != DEFENSIVE
                or (weights.loc[:, [s for s in law.UNIVERSE if s not in allowed]] > 1e-10)
                .any()
                .any()
            ):
                raise QuantError(
                    "Require actual styles and all fixed defensive buckets, not names."
                )
            families[key], defensive[key], paths[key] = used, buckets, receipt
    targets = source.source_frame(native)
    expected = pd.DatetimeIndex([day for day in data.close.index if is_month_end(day)])
    if (
        not targets.index.equals(data.close.index)
        or tuple(targets.columns) != law.UNIVERSE
        or not targets.dropna(how="all").index.equals(expected)
        or proof["native_month_end_opinions"] != len(expected)
    ):
        raise QuantError("Keep the whole fixed-policy native lattice without invented warmup.")
    shadow = _frame(_artifact(paths["10y-base"], "account.csv"))
    expert = ArchivedMonthlyExpert(
        ExpertIdentity(law.ID, MECHANISM, registered[0]["at"]), targets, shadow, law.UNIVERSE
    )
    anchors = [previous_session(window["start"]) for window in protocol["windows"]]
    dates = sorted(
        set([*expected, *anchors, shadow.index[124], shadow.index[125], shadow.index[-1]])
    )
    active = targets.dropna(how="all")
    for day in dates:
        observed, known = expert.observe(day, 126), active.loc[:day].tail(1)
        if (
            known.empty
            or observed.target is None
            or observed.target_known_at != known.index[0]
            or observed.target_known_at > day
            or not observed.target.equals(known.iloc[0])
            or not observed.risk_free.equals(data.risk_free.loc[observed.risk_free.index])
        ):
            raise QuantError("Native availability, exact common RF or known opinion differs.")
    boundary_days = [anchors[0], shadow.index[124], shadow.index[125]]
    boundary = [expert.observe(day, 126) for day in boundary_days]
    if [len(row.net_returns) for row in boundary] != [0, 125, 126] or [
        row.score_mature for row in boundary
    ] != [False, False, True]:
        raise QuantError("Do not confuse fixed-target availability with mature NET history.")
    return expert, {
        "candidate_id": law.ID,
        "mechanism_group": MECHANISM,
        "adapter": "monthly",
        "source_packet_schema": 3,
        "introduced_at": registered[0]["at"],
        "registered_event_sha256": registered[0]["sha256"],
        "financial_reviewed_at": reviewed[0]["at"],
        "registration": source.reference(source.PACKET),
        "financial_results": source.reference(source.RESULTS),
        "native_source_targets": native,
        "completed_paths": result["review"]["paths"],
        "actual_style_families_by_path": families,
        "actual_fixed_defensive_buckets_by_path": defensive,
        "native_month_end_opinions": len(expected),
        "availability_observations_checked": len(dates),
        "NET_maturity_boundary_observations": [
            {
                "date": str(day.date()),
                "NET_observations": len(row.net_returns),
                "score_mature": row.score_mature,
            }
            for day, row in zip(boundary_days, boundary, strict=True)
        ],
        "native_RF_exact_common": True,
        "funding_and_independent_BT_checked_all_four_paths": True,
        "original_source_scope_and_information_clock_preserved": True,
        "source_targets_shadows_or_accounts_replayed": False,
        "standalone_financial_verdict_preserved": result["review"]["status"],
        "technical_admission_is_final_qualification": False,
        "current_global_pool_or_roster_changed": False,
        "pool_inclusion_requires_new_separate_study": True,
        "historical_counterfactual_availability": True,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def load_admitted_expert(admission_reference, universe, risk_free):
    if (
        universe != law.UNIVERSE
        or not isinstance(risk_free, pd.Series)
        or not isinstance(risk_free.index, pd.DatetimeIndex)
        or not risk_free.index.is_unique
        or not risk_free.index.is_monotonic_increasing
        or risk_free.index.tz is not None
    ):
        raise QuantError("This source Adapter cannot remap or crop the admitted asset universe.")
    admission = read_json(base.checked_reference(admission_reference))
    if (
        admission.get("technical_expert_admitted") is not True
        or admission.get("global_pool_changed") is not False
        or type(admission.get("new_strategy_evaluations")) is not int
        or admission.get("new_strategy_evaluations") != 0
        or admission.get("order_authority") is not False
    ):
        raise QuantError("Require actual technical admission, not pool or trading authority.")
    freeze = read_json(base.checked_reference(admission["review_implementation_freeze"]))
    required = set(OWN)
    if not required.issubset(
        {ref["path"] for ref in freeze["implementation_and_test_inputs"]}
    ) or not required.issubset({ref["path"] for ref in freeze["complete_dependency_inputs"]}):
        raise QuantError("The admission needs the actual frozen review and dependency closure.")
    for reference in (
        freeze["implementation_and_test_inputs"] + freeze["complete_dependency_inputs"]
    ):
        base.checked_reference(reference)
    expert, evidence = inspect_source(read_json(POLICY))
    admitted_at, frozen_at = [
        pd.Timestamp(value) for value in (admission["admitted_at"], freeze["created_at"])
    ]
    if (
        pd.isna(admitted_at)
        or admitted_at.tzinfo is None
        or pd.isna(frozen_at)
        or frozen_at.tzinfo is None
        or admitted_at < pd.Timestamp(evidence["financial_reviewed_at"])
        or admitted_at < frozen_at
    ):
        raise QuantError("Actual technical admission cannot predate its frozen review or source.")
    full = expert.observe(pd.Timestamp("2026-10-05"), 2512)
    if (
        admission["evidence"] != evidence
        or len(full.risk_free) != 2512
        or not full.risk_free.index.isin(risk_free.index).all()
        or not full.risk_free.equals(risk_free.loc[full.risk_free.index])
    ):
        raise QuantError("Admitted source evidence or consumer assessment RF changed.")
    return expert, {**evidence, "actual_expert_admitted_at": admission["admitted_at"]}
