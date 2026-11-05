from __future__ import annotations

import json
import math
from types import FunctionType

import numpy as np
import pandas as pd

from us_quant import ssd_style_activity_financial_v1 as source
from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.config import QuantError
from us_quant.fixed_four_sleeve_admission_v1 import load_admitted_expert as _consume_saved
from us_quant.monthly_style_reversal_admission_v1 import _artifact, _frame
from us_quant.storage import digest_json, read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity, StrategyExpert

base, law = source.base, source.law
POLICY = source.ROOT / "config/style-activity-admission-v1.json"
FOLDER = base.REPORTS / "style-activity-admission-20261105-v1"
REVIEW_FREEZE = FOLDER / "review-implementation-freeze-v1.json"
ADMISSION = FOLDER / "technical-admission-v1.json"
PUBLIC = source.ROOT / "evidence/style_activity_financial_20261105_admission_v1.json"
MECHANISM = "cross_sectional_reported_activity_participation"
DEFENSIVE = ("BIL",)
OWN = (
    "config/style-activity-admission-v1.json",
    "src/us_quant/style_activity_admission_v1.py",
    "tests/test_style_activity_admission_v1.py",
    "STYLE_ACTIVITY_ADMISSION_V1.md",
)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_style_activity_technical_admission_20261105_v1",
        "candidate_id": law.ID,
        "mechanism_group": MECHANISM,
        "source_packet_schema": 3,
        "universe": list(law.UNIVERSE),
        "admitted_activity_symbols": list(law.CORE),
        "minimum_actual_style_families_per_path": 3,
        "required_actual_defensive_assets": list(DEFENSIVE),
        "source_price_warmup_position": 64,
        "source_input_lag_sessions": 1,
        "binary64_input_values_per_native_opinion": 760,
        "existing_positive_weight_threshold": 1e-10,
        "existing_independent_BT_tolerance_usd": 1e-6,
        "maximum_gross_exposure": 1.0,
        "minimum_cash_usd": 0.0,
        "score_lookback": 126,
        "RF_comparison": "exact_original_canonical_review_market_series",
        "old_activity_source_RF_imported_or_normalized": False,
        "financial_performance_is_technical_admission_gate": False,
        "new_source_requests": 0,
        "source_targets_or_accounts_replayed": False,
        "global_pool_promotion": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError(
            "Preserve original technical standards and the exact four-fund activity scope."
        )


def check_trace(data, targets, trace_reference):
    active = targets.dropna(how="all")
    values_checked = 0
    with base.checked_reference(trace_reference).open() as stream:
        for day in active.index:
            before, after = json.loads(stream.readline()), json.loads(stream.readline())
            position = int(data.close.index.get_loc(day))
            expected = {
                symbol: {
                    key: [float(v).hex() for v in sample]
                    for key, sample in zip(
                        ("adjusted_closes", "reported_closes", "reported_share_volumes"),
                        (
                            data.close[symbol].iloc[position - 64 : position],
                            data.raw_close[symbol].iloc[position - 63 : position],
                            data.volume[symbol].iloc[position - 63 : position],
                        ),
                        strict=True,
                    )
                }
                for symbol in law.CORE
            }
            if (
                before
                != {
                    "record": "decision_input",
                    "decision_session": str(day.date()),
                    "information_clock": law.CLOCK,
                    "activity_symbols": list(law.CORE),
                    "activity_session_dates": data.close.index[position - 63 : position]
                    .strftime("%Y-%m-%d")
                    .tolist(),
                    "inputs_binary64_hex": expected,
                    "decision_session_own_or_future_bar_used": False,
                }
                or after["record"] != "decision_resolved"
                or after["asset_order"] != list(law.UNIVERSE)
                or after["complete_target_binary64_hex"]
                != [float(v).hex() for v in active.loc[day]]
                or after["audit"]["decision_session"] != str(day.date())
                or after["audit"]["decision_information_clock"] != law.CLOCK
                or after["audit"]["first_prior_adjusted_close"]
                != str(data.close.index[position - 64].date())
                or after["audit"]["first_activity_session"]
                != str(data.close.index[position - 63].date())
                or after["audit"]["last_input_session"]
                != str(data.close.index[position - 1].date())
                or after["audit"]["decision_session_own_or_future_bar_used"] is not False
                or after["audit"]["price_warmup_implies_mature_NET_score"] is not False
                or after["audit"]["counterfactual_historical_availability"] is not True
                or after["audit"]["actual_policy_introduction_or_admission_backdated"] is not False
            ):
                raise QuantError(
                    "Retained exact source inputs, clocks or complete targets changed."
                )
            audit = after["audit"]
            balances = audit["activity_balances"]
            if (
                set(balances) != set(law.CORE)
                or any(not math.isfinite(v) or not -1 <= v <= 1 for v in balances.values())
                or set(audit["ranking_keys"]) != set(law.CORE)
                or any(not math.isfinite(v) for v in audit["ranking_keys"].values())
                or audit["return_and_activity_observations"] != 63
                or audit["invalid_input_defaulted_to_cash"] is not False
                or audit["reported_activity_is_actual_flow_trade_direction_or_market_breadth"]
                is not False
            ):
                raise QuantError(
                    "Saved defined activity assessments cannot become flows or fallbacks."
                )
            selected = audit["selected_assets"]
            held = active.loc[day]
            nonzero = held[held != 0]
            if (
                type(audit["strictly_positive_style_count"]) is not int
                or not 0 <= audit["strictly_positive_style_count"] <= 4
                or not (
                    (selected == ["BIL"] and audit["strictly_positive_style_count"] < 3)
                    or (
                        len(selected) == len(set(selected)) == 3
                        and set(selected).issubset(law.CORE)
                        and audit["strictly_positive_style_count"] >= 3
                    )
                )
                or set(nonzero.index) != set(selected)
                or not (nonzero == 0.98 / len(selected)).all()
            ):
                raise QuantError(
                    "The saved complete allocation cannot change its recorded asset budget."
                )
            values_checked += 760
        if stream.readline():
            raise QuantError(
                "Retain exactly one saved input and one result per native source opinion."
            )
    return values_checked


def inspect_source(policy: dict) -> tuple[StrategyExpert, dict]:
    validate(policy)
    data, frozen, _ = source.inputs()
    packet = source.checked_packet(frozen)
    result = read_json(source.RESULTS)
    registered, reviewed, proved = [
        source.events(k) for k in ("registered", "review", "target_proof")
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
        or proved[0]["body"]["real_main_capital_prefix_audit"]
        != result["real_main_capital_prefix_audit"]
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
        raise QuantError("Require one unchanged registered/published complete source identity.")
    native = result["native_source_proof"]
    proof = read_json(base.checked_reference(native["proof"]))
    started = read_json(base.checked_reference(proof["source_start"]))
    if (
        proof["frozen_specification_sha256"] != digest_json(frozen)
        or proof["publication_event_sha256"] != published["sha256"]
        or started["publication_event_sha256"] != published["sha256"]
        or proof["real_native_indicators_first_generated_after_actual_publication"] is not True
        or proof["real_native_targets_first_generated_after_actual_publication"] is not True
        or proof["all_independent_signed_activity_targets_exact"] is not True
        or proof["full_MarketData_four_fund_activity_scope_preserved"] is not True
        or proof["old_activity_source_RF_imported_or_normalized"] is not False
        or proof["decision_day_or_future_input_used"] is not False
        or type(proof["source_price_volume_binary64_inputs_checked"]) is not int
        or type(proof["independent_four_style_balance_checks"]) is not int
        or type(proof["maximum_independent_scalar_numpy_balance_gap"]) is not float
        or not math.isfinite(proof["maximum_independent_scalar_numpy_balance_gap"])
        or not 0 <= proof["maximum_independent_scalar_numpy_balance_gap"] <= 1e-12
        or proof["whole_native_prehistory_retained"] is not True
        or proof["synthetic_initial_recommendation_added"] is not False
        or not pd.Timestamp(published["at"])
        <= pd.Timestamp(started["started_at"])
        <= pd.Timestamp(proof["checked_at"])
        < pd.Timestamp(reviewed[0]["at"])
    ):
        raise QuantError("Preserve the actual published four-field proof without source replay.")
    prefixes = read_json(base.checked_reference(result["real_main_capital_prefix_audit"]))
    if (
        prefixes["auxiliary_main_book_audit_runs"] != 5
        or prefixes["extra_economic_rules_or_qualifying_windows"] != 0
        or len(prefixes["prefixes"]) != 5
        or [r["formal_return_position"] for r in prefixes["prefixes"]]
        != [125, 503, 1006, 1509, 2012]
        or any(r["capital_and_decisions_exact"] is not True for r in prefixes["prefixes"])
    ):
        raise QuantError(
            "Keep all five exact saved main-book prefixes, not new qualifying windows."
        )
    for row in prefixes["prefixes"]:
        for reference in row["artifacts"]:
            base.checked_reference(reference)
    if set(result["review"]["paths"]) != base.PATH_KEYS:
        raise QuantError("Keep all four actual original paths before expert admission.")
    starts, completes = [source.events(k) for k in ("path_started", "path_completed")]
    if (
        len(starts) != 4
        or len(completes) != 4
        or {r["body"]["path_key"] for r in starts} != base.PATH_KEYS
        or {r["body"]["path_key"] for r in completes} != base.PATH_KEYS
    ):
        raise QuantError("Require one real start/completion per original path.")
    paths, families, defensive = {}, {}, {}
    protocol = read_json(base.PROTOCOL)
    allowed = (*law.CORE, *DEFENSIVE)
    for window in protocol["windows"]:
        dates = sessions(window["start"], window["end"])
        for scenario in ("base", "stress"):
            key = f"{window['years']}y-{scenario}"
            reference = result["review"]["paths"][key]
            receipt = base.verify_path(reference, frozen, law.ID, key)
            begin = next(r for r in starts if r["body"]["path_key"] == key)
            end = next(r for r in completes if r["body"]["path_key"] == key)
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
                or not all(
                    math.isfinite(v)
                    for v in (
                        receipt["execution"]["minimum_cash_usd"],
                        receipt["execution"]["maximum_gross_exposure"],
                        receipt["independent_bt_max_equity_difference_usd"],
                    )
                )
                or receipt["execution"]["minimum_cash_usd"] < 0
                or not 0 <= receipt["execution"]["maximum_gross_exposure"] <= 1
                or not 0 <= receipt["independent_bt_max_equity_difference_usd"] <= 1e-6
                or receipt["broker_orders"] != 0
            ):
                raise QuantError("Invalid source funding/accounting cannot become an expert.")
            account, weights = [
                _frame(_artifact(receipt, name)) for name in ("account.csv", "weights.csv")
            ]
            needed = ("equity", "return", "risk_free", "cash", "gross_exposure", "cost", "orders")
            if (
                not account.index.equals(dates)
                or not weights.index.equals(dates)
                or tuple(weights.columns) != law.UNIVERSE
                or not set(needed).issubset(account.columns)
                or not np.isfinite(account.loc[:, list(needed)].to_numpy()).all()
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
                raise QuantError(
                    "Do not crop source dates, borrow, remap assets or replace native RF."
                )
            used = [s for s in law.CORE if (weights[s] > 1e-10).any()]
            buckets = [s for s in DEFENSIVE if (weights[s] > 1e-10).any()]
            if (
                len(used) < 3
                or tuple(buckets) != DEFENSIVE
                or (weights.loc[:, [s for s in law.UNIVERSE if s not in allowed]] > 1e-10)
                .any()
                .any()
            ):
                raise QuantError("Require actual admitted styles/BIL, not nominal coverage.")
            paths[key], families[key], defensive[key] = receipt, used, buckets
    targets = source.source_frame(native)
    expected = pd.DatetimeIndex(
        [
            day
            for position, day in enumerate(data.close.index)
            if position >= 64 and is_month_end(day)
        ]
    )
    positions = sorted({0, 6, 12, len(expected) // 2, len(expected) - 1})
    if (
        not targets.index.equals(data.close.index)
        or tuple(targets.columns) != law.UNIVERSE
        or not targets.dropna(how="all").index.equals(expected)
        or proof["native_month_end_opinions"] != len(expected)
        or proof["source_price_volume_binary64_inputs_checked"] != len(expected) * 760
        or proof["independent_four_style_balance_checks"] != len(expected) * 4
        or proof["prefix_truncations"] != [str(expected[p].date()) for p in positions]
    ):
        raise QuantError("Keep full64-position lagged warmup and every genuine monthly opinion.")
    checked_values = check_trace(data, targets, proof["retained_input_resolved_trace"])
    if checked_values != proof["source_price_volume_binary64_inputs_checked"]:
        raise QuantError("Actual retained inputs must match their saved independent source proof.")
    shadow_ref = _artifact(paths["10y-base"], "account.csv")
    shadow = _frame(shadow_ref)
    expert = ArchivedMonthlyExpert(
        ExpertIdentity(law.ID, MECHANISM, registered[0]["at"]), targets, shadow, law.UNIVERSE
    )
    anchors = [previous_session(window["start"]) for window in protocol["windows"]]
    dates = sorted(
        set(
            [
                *expected,
                *anchors,
                data.close.index[63],
                previous_session(expected[0]),
                shadow.index[124],
                shadow.index[125],
                shadow.index[-1],
            ]
        )
    )
    active = targets.dropna(how="all")
    for day in dates:
        observed, known = expert.observe(day, 126), active.loc[:day].tail(1)
        if known.empty:
            if observed.target is not None or observed.target_known_at is not None:
                raise QuantError("Unavailable warmup cannot receive an invented opinion.")
        elif (
            observed.target is None
            or observed.target_known_at != known.index[0]
            or observed.target_known_at > day
            or not observed.target.equals(known.iloc[0])
        ):
            raise QuantError("Native availability or exact known opinion differs.")
        if not observed.risk_free.equals(data.risk_free.loc[observed.risk_free.index]):
            raise QuantError("Preserve exact current native RF without an assessment bridge.")
    boundary_days = [anchors[0], shadow.index[124], shadow.index[125]]
    boundary = [expert.observe(day, 126) for day in boundary_days]
    if [len(r.net_returns) for r in boundary] != [0, 125, 126] or [
        r.score_mature for r in boundary
    ] != [False, False, True]:
        raise QuantError("Price warmup and target availability are not mature NET scores.")
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
        "target_reference": native["targets"],
        "shadow_reference": shadow_ref,
        "native_source_targets": native,
        "completed_paths": result["review"]["paths"],
        "actual_style_families_by_path": families,
        "actual_defensive_assets_by_path": defensive,
        "native_month_end_opinions": len(expected),
        "source_price_warmup_position": 64,
        "unavailable_before_price_warmup_checked": True,
        "source_price_volume_binary64_inputs_checked": checked_values,
        "independent_four_style_balance_checks": len(expected) * 4,
        "saved_main_capital_prefixes_exact": 5,
        "full_OHLCV_independent_source_proof_preserved": True,
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
        "old_activity_source_RF_imported_or_normalized": False,
        "funding_and_independent_BT_checked_all_four_paths": True,
        "original_source_scope_and_information_clock_preserved": True,
        "source_targets_shadows_or_accounts_replayed": False,
        "standalone_financial_verdict_preserved": result["review"]["status"],
        "technical_admission_is_final_qualification": False,
        "current_global_pool_or_roster_changed": False,
        "pool_inclusion_requires_new_separate_study": True,
        "related_price_activity_family_is_new_independent_risk": False,
        "historical_counterfactual_availability": True,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def load_admitted_expert(admission_reference, universe, risk_free):
    return FunctionType(
        _consume_saved.__code__,
        {**_consume_saved.__globals__, **globals()},
        argdefs=_consume_saved.__defaults__,
    )(admission_reference, universe, risk_free)
