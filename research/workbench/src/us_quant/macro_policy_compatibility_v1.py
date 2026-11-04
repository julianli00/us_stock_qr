from __future__ import annotations

import argparse
import ast
import json
import os
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import legacy_defensive_rotation_expert_v1 as precedent
from us_quant import legacy_real_yield_expert_eligibility_v1 as original_review
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_seven_expert_meta_v1 as helpers
from us_quant.calendar import is_month_end, previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.storage import (
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_text_atomic,
)
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity, _ArchivedExpert

ROOT, BASE = helpers.ROOT, helpers.base
FOLDER = BASE.REPORTS / "macro-policy-compatibility-20261103-v1"
CONTRACT = FOLDER / "compatibility-contract-v1.json"
INTENT = FOLDER / "implementation-intent-v1.json"
OUTPUT = FOLDER / "compatibility-result-v1.json"
VIEWS = ROOT / "data/macro-policy-assessment-compatibility-20261103-v1"
CANDIDATE = "macro_real_yield_factor_tilt"
GROUP = "real_discount_rate_factor_composition"
PATHS = ("10y-base", "10y-stress", "5y-base", "5y-stress")
OWN = (
    "src/us_quant/macro_policy_compatibility_v1.py",
    "tests/test_macro_policy_compatibility_v1.py",
    "tests/conftest.py",
)


def frame(ref, *, round_trip=True):
    kwargs = {"float_precision": "round_trip"} if round_trip else {}
    return pd.read_csv(BASE.checked_reference(ref), index_col=0, parse_dates=True, **kwargs)


class MacroSeedAwarePolicyOpinion(_ArchivedExpert):
    """Observe a recorded policy opinion; emission and funding events remain separate."""

    def __init__(self, identity, native_targets, assessment_shadow, universe):
        if identity.identifier != CANDIDATE or set(native_targets) != set(PATHS):
            raise QuantError("This compatibility contract covers only the declared macro source.")
        canonical = native_targets["10y-base"]
        super().__init__(identity, canonical, assessment_shadow, universe)
        self._native_variants = {key: value.copy() for key, value in native_targets.items()}
        active = canonical.dropna(how="all")
        genuine = active.loc[[is_month_end(day) for day in active.index]]
        if genuine.empty:
            raise QuantError("A capital seed cannot manufacture a policy recommendation history.")
        expected = pd.DatetimeIndex(
            [day for day in canonical.loc[genuine.index[0] :].index if is_month_end(day)]
        )
        if not genuine.index.equals(expected):
            raise QuantError("Every original genuine monthly policy recommendation is required.")
        self._policy_rows = genuine
        self._initialization_roles = []
        for key, targets in self._native_variants.items():
            if (
                not targets.index.equals(canonical.index)
                or not targets.columns.equals(canonical.columns)
                or (targets.isna().any(axis=1) & ~targets.isna().all(axis=1)).any()
            ):
                raise QuantError("Keep each full original native request matrix without remapping.")
            actual = targets.dropna(how="all")
            month = actual.loc[[is_month_end(day) for day in actual.index]]
            if not month.equals(genuine):
                raise QuantError(
                    "Native policy rows must match exactly across all four capital paths."
                )
            anchor = previous_session("2016-10-06" if key.startswith("10y") else "2021-10-06")
            extras = actual.loc[[not is_month_end(day) for day in actual.index]]
            if anchor > targets.index[-1]:
                if not extras.empty:
                    raise QuantError(
                        "Future funding metadata cannot validate another nonmonthly event."
                    )
                continue
            known = genuine.loc[genuine.index < anchor].tail(1)
            if (
                len(extras) != 1
                or extras.index[0] != anchor
                or known.empty
                or not extras.iloc[0].equals(known.iloc[0])
            ):
                raise QuantError(
                    "A nonmonthly policy event is not a proven capital initialization seed."
                )
            self._initialization_roles.append(
                {
                    "path_key": key,
                    "original_capital_initialization_date": str(anchor.date()),
                    "original_policy_row_known_at": str(known.index[0].date()),
                    "seed_kept_in_native_matrix": True,
                    "seed_backdated_or_relabelled_policy_emission": False,
                }
            )

    def _known_target(self, as_of):
        known = self._policy_rows.loc[:as_of]
        if known.empty:
            return None, None
        return known.iloc[-1].copy(), known.index[-1]

    def event_context(self, as_of):
        day = pd.Timestamp(as_of)
        if day not in self._targets.index:
            raise QuantError("Event context requires an original covered source session.")
        _, known = self._known_target(day)
        seeds = [
            row
            for row in self._initialization_roles
            if row["original_capital_initialization_date"] == str(day.date())
        ]
        emitted = day in self._policy_rows.index
        return {
            "as_of": str(day.date()),
            "original_policy_opinion_known_at": str(known.date()) if known is not None else None,
            "native_policy_recommendation_emitted_today": emitted,
            "native_capital_initialization_events": seeds,
            "no_new_native_source_request_today": not emitted and not seeds,
            "observation_is_a_new_rebalance_request": False,
            "opinion_copied_from_actual_source_positions": False,
            "seed_removed_or_retimed_to_month_end": False,
            "all_native_seed_and_request_rows_retained": True,
        }

    def preservation(self):
        return {
            "native_full_active_rows_by_path": {
                key: len(value.dropna(how="all")) for key, value in self._native_variants.items()
            },
            "genuine_policy_rows": len(self._policy_rows),
            "original_policy_recommendation_dates": self._policy_rows.index.strftime(
                "%Y-%m-%d"
            ).tolist(),
            "initialization_event_roles": self._initialization_roles,
            "seed_stripped_nominal_view_artifacts": 0,
            "opinion_observation_is_source_trade_or_rebalance_instruction": False,
        }


def assessment_view(native_shadow, native_source_default_rf, common_default_rf):
    result = precedent.assessment_view(native_shadow, native_source_default_rf, common_default_rf)
    if not result["risk_free"].equals(common_default_rf.loc[result.index]):
        raise QuantError(
            "The explicit assessment RF view must equal its exact pinned common reader."
        )
    unchanged = [name for name in native_shadow if name != "risk_free"]
    if not result[unchanged].equals(native_shadow[unchanged]):
        raise QuantError("No source NET return,NAV,cash,fee or exposure can change.")
    return result


def dependencies():
    return FunctionType(
        helpers.admitted.import_template.dependencies.__code__,
        {**globals(), "ast": ast, "prior": helpers, "base": BASE},
    )()


def source_refs():
    old = read_json(original_review.INTENT)
    proof = read_json(ROOT / "evidence/legacy_defensive_rotation_expert_20261030_admission_v1.json")
    admission = read_json(BASE.checked_reference(proof["private_admission"]))
    expert = admission["expert"]
    refs = [
        *old["source_references"],
        *old["implementation_and_test_inputs"],
        BASE.reference(original_review.INTENT),
        BASE.reference(original_review.OUTPUT),
        BASE.reference(original_review.FOLDER / "completion-v1.json"),
        BASE.reference(
            ROOT / "evidence/legacy_defensive_rotation_expert_20261030_admission_v1.json"
        ),
        proof["private_admission"],
        expert["target_reference"],
        expert["shadow_reference"],
        expert["native_target_reference"],
        expert["native_shadow_reference"],
        expert["view_provenance"],
        expert["common_assessment_RF_reference"],
        BASE.reference(CONTRACT),
        BASE.reference(FOLDER / "readonly-source-role-preflight-v1.json"),
        BASE.reference(FOLDER / "actual-coordinator-scope-v1.json"),
    ]
    return helpers.unique_refs(refs)


def freeze():
    lifecycle.runtime_guard()
    os.umask(0o077)
    if INTENT.exists():
        raise QuantError("Preserve the complete or partial first compatibility intent.")
    with lifecycle.research_job():
        if (
            file_digest(CONTRACT)
            != "10efa67ad49d2523e3eb6da93b930eb8b3041046888e6f06bdd0c4802f23a244"
        ):
            raise QuantError("Keep the separately frozen coordinator role contract.")
        xml = FOLDER / "offline-tests-v1.xml"
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        if not suites or any(
            int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
        ):
            raise QuantError(
                "Complete role,event,assessment,missingness and prefix regression checks."
            )
        proof = read_json(FOLDER / "validation-source-v1.json")
        own = [BASE.reference(ROOT / p) for p in OWN]
        own.append(BASE.reference(ROOT / "MACRO_POLICY_COMPATIBILITY_V1.md"))
        outputs = [BASE.reference(xml), BASE.reference(FOLDER / "offline-tests-v1.log")]
        if (
            proof["implementation_and_test_inputs"] != own
            or proof["outputs"] != outputs
            or proof["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or proof["exit_code"] != 0
        ):
            raise QuantError(
                "Bind the successful actual compatibility implementation and reading convention."
            )
        refs = source_refs()
        for row in refs:
            BASE.checked_reference(row)
        history = BASE.journal_events(BASE.JOURNAL)
        if len(history) != 722:
            raise QuantError(
                "No unrelated financial state may enter this finite compatibility node."
            )
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "contract": BASE.reference(CONTRACT),
                "implementation_dependencies": [
                    *dependencies(),
                    BASE.reference(ROOT / "MACRO_POLICY_COMPATIBILITY_V1.md"),
                ],
                "source_references": refs,
                "validation_refs": [*outputs, BASE.reference(FOLDER / "validation-source-v1.json")],
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "actual_derived_view_or_compatibility_probes_created": False,
                "new_financial_trials_or_pool_change": 0,
                "no_outbound_network_required": True,
            },
        )
    return BASE.reference(INTENT)


def old_defensive_probe(universe, as_ofs):
    public = read_json(
        ROOT / "evidence/legacy_defensive_rotation_expert_20261030_admission_v1.json"
    )
    record = read_json(BASE.checked_reference(public["private_admission"]))["expert"]
    expert = ArchivedMonthlyExpert(
        ExpertIdentity(record["id"], record["mechanism_group"], record["introduced_at"]),
        frame(record["target_reference"]),
        frame(record["shadow_reference"]),
        universe,
    )
    return expert, {str(day.date()): expert.observe(day, 126) for day in as_ofs}


def same_observation(left, right):
    return (
        left.identity == right.identity
        and left.as_of == right.as_of
        and left.target_known_at == right.target_known_at
        and left.score_mature == right.score_mature
        and (
            left.target is None
            and right.target is None
            or left.target is not None
            and right.target is not None
            and left.target.equals(right.target)
        )
        and left.net_returns.equals(right.net_returns)
        and left.risk_free.equals(right.risk_free)
    )


def probe():
    lifecycle.runtime_guard()
    os.umask(0o077)
    if OUTPUT.exists() or (FOLDER / "attempt-v1.json").exists():
        raise QuantError("Do not replay a completed or partial compatibility identity.")
    intent = read_json(INTENT)
    for row in (
        intent["source_references"]
        + intent["implementation_dependencies"]
        + intent["validation_refs"]
    ):
        BASE.checked_reference(row)
    if intent["package_versions"] != [
        list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())
    ]:
        raise QuantError("The frozen isolated compatibility environment changed.")
    with lifecycle.research_job():
        before = helpers.current.current_counts(BASE.journal_events(BASE.JOURNAL))
        write_new_receipt(
            FOLDER / "attempt-v1.json", {"started_at": utc_now(), "intent": BASE.reference(INTENT)}
        )
        try:
            snapshot = read_json(original_review.SNAPSHOT)
            bundle = read_json(
                ROOT / "reports/macro-tilt-20261010/macro_real_yield_factor_tilt/bundle.json"
            )
            original_review.check_identity(
                snapshot,
                read_json(ROOT / "evidence/macro_tilt_20261010_registration.json"),
                read_json(ROOT / "evidence/macro_tilt_20261010_results.json"),
                bundle,
                read_json(ROOT / "config/macro-factor-tilt.json"),
            )
            native_targets = {
                f"{r['years']}y-{r['scenario']}": frame(r["targets"]) for r in bundle["paths"]
            }
            native = frame(
                next(r for r in bundle["paths"] if (r["years"], r["scenario"]) == (10, "base"))[
                    "strategy"
                ]
            )
            native_rf = frame(
                snapshot["specification"]["market"]["risk_free"], round_trip=False
            ).iloc[:, 0]
            approved = read_json(
                BASE.checked_reference(
                    read_json(
                        ROOT
                        / "evidence/legacy_defensive_rotation_expert_20261030_admission_v1.json"
                    )["private_admission"]
                )
            )
            common_ref = approved["expert"]["common_assessment_RF_reference"]
            common_rf = frame(common_ref, round_trip=False).iloc[:, 0]
            universe = (
                "SPY",
                "IEF",
                "GLD",
                "BIL",
                "MTUM",
                "VLUE",
                "QUAL",
                "USMV",
                "TLT",
                "QQQ",
                "XLK",
                "SOXX",
                "IJR",
                "PKW",
            )
            asofs = [
                pd.Timestamp(d)
                for d in (
                    "2016-10-05",
                    "2016-10-18",
                    "2016-10-31",
                    "2017-04-28",
                    "2021-10-05",
                    "2026-10-05",
                )
            ]
            existing, old_observations = old_defensive_probe(universe, asofs)
            view = assessment_view(native, native_rf, common_rf)
            new_output_directory(VIEWS)
            path = VIEWS / "native-NET-common-assessment-RF.csv"
            write_text_atomic(path, view.to_csv(float_format="%.17g"))
            stored = frame(BASE.reference(path))
            unchanged = [col for col in native if col != "risk_free"]
            if not stored[unchanged].equals(native[unchanged]) or not stored["risk_free"].equals(
                common_rf.loc[stored.index]
            ):
                raise QuantError(
                    "Stored assessment view changed source money or the exact common RF."
                )
            identity = ExpertIdentity(CANDIDATE, GROUP, intent["frozen_at"])
            expert = MacroSeedAwarePolicyOpinion(identity, native_targets, stored, universe)
            observations = []
            for day in asofs:
                row = expert.observe(day, 126)
                if row.target is None or not row.risk_free.equals(
                    common_rf.loc[row.risk_free.index]
                ):
                    raise QuantError("Cached policy-opinion availability or assessment RF failed.")
                if not same_observation(
                    old_observations[str(day.date())], existing.observe(day, 126)
                ):
                    raise QuantError("An existing successfully admitted expert's behavior changed.")
                observations.append(
                    {
                        **expert.event_context(day),
                        "NET_observations": len(row.net_returns),
                        "score_mature": row.score_mature,
                        "target_known_at": str(row.target_known_at.date()),
                    }
                )
            prefixes = []
            for day in (
                pd.Timestamp("2016-10-31"),
                pd.Timestamp("2017-04-28"),
                pd.Timestamp("2021-10-05"),
            ):
                early = MacroSeedAwarePolicyOpinion(
                    identity,
                    {k: t.loc[:day] for k, t in native_targets.items()},
                    stored.loc[:day],
                    universe,
                )
                if not same_observation(
                    early.observe(day, 126), expert.observe(day, 126)
                ) or early.event_context(day) != expert.event_context(day):
                    raise QuantError(
                        "Future native roles or assessment data changed an observed prefix."
                    )
                prefixes.append(str(day.date()))
            after = helpers.current.current_counts(BASE.journal_events(BASE.JOURNAL))
            if before != after:
                raise QuantError("Technical compatibility cannot create a financial trial.")
            for ref in intent["source_references"]:
                BASE.checked_reference(ref)
            report = {
                "completed_at": utc_now(),
                "compatibility_id": "ssd_macro_policy_compatibility_20261103_v1",
                "status": "compatible_scoped_policy_opinions_with_explicit_assessment_view",
                "intent": BASE.reference(INTENT),
                "contract": BASE.reference(CONTRACT),
                "native_role_preservation": expert.preservation(),
                "cached_observation_checks": observations,
                "prefix_truncations": prefixes,
                "old_defensive_expert_regression_checks": len(asofs),
                "existing_strategy_experts_v1_code_unchanged": True,
                "native_source_RF_still_not_exact_common": not native["risk_free"].equals(
                    common_rf.loc[native.index]
                ),
                "native_RF_vs_common_different_cells": int(
                    (native["risk_free"] != common_rf.loc[native.index]).sum()
                ),
                "explicit_new_assessment_view": BASE.reference(path),
                "native_NET_NAV_cash_fees_orders_gross_turnover_and_dates_exactly_unchanged": True,
                "common_RF_reference": common_ref,
                "common_RF_reader": "original_pandas_default",
                "native_market_RF_reader": "original_pandas_default",
                "native_book_and_new_view_reader": "pandas_round_trip",
                "assessment_only_RF_changed_no_round_isclose_or_tolerance": True,
                "new_nominal_target_or_seed_removed_view_artifacts": 0,
                "prior_direct_no_view_qualification_still_blocked_and_unchanged": True,
                "old_financial_verdict_or_CAGR_Sharpe_recomputed_or_changed": False,
                "future_scoring_or_financial_effect_absent_claimed": False,
                "old2NYSE7day_limited_source_vintage_PIT_scope_unchanged": True,
                "new_expert_admitted_or_9th_representative_or_pool_expansion": False,
                "future_financial_use_requires_separate_frozen_identity": True,
                "new_source_requests_shadow_main_benchmark_or_financial_trials": 0,
                "actual_financial_counts_unchanged": before,
                "investment_objective_verified": False,
                "order_authority": False,
            }
            write_new_receipt(
                VIEWS / "assessment-provenance-v1.json",
                {
                    "created_at": utc_now(),
                    "compatibility_intent": BASE.reference(INTENT),
                    "native_book_reference": next(
                        r["strategy"]
                        for r in bundle["paths"]
                        if (r["years"], r["scenario"]) == (10, "base")
                    ),
                    "native_market_RF_reference": snapshot["specification"]["market"]["risk_free"],
                    "native_RF_reader": "original_pandas_default",
                    "common_RF_reference": common_ref,
                    "common_RF_reader": "original_pandas_default",
                    "assessment_view_reference": BASE.reference(path),
                    "stored_view_reader": "pandas_round_trip",
                    "NET_NAV_cash_costs_orders_gross_turnover_dates_preserved_exactly": True,
                    "source_native_RF_still_unequal_common": report[
                        "native_source_RF_still_not_exact_common"
                    ],
                    "future_scoring_or_financial_effect_absent_claimed": False,
                    "nominal_seed_stripped_target_view_created": False,
                },
            )
            report["assessment_view_provenance"] = BASE.reference(
                VIEWS / "assessment-provenance-v1.json"
            )
            write_new_receipt(OUTPUT, report)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(error),
                    "partial_evidence_preserved": True,
                    "no_policy_position_copy_or_new_target_fallback": True,
                    "new_financial_trials": 0,
                },
            )
            raise
    return BASE.reference(OUTPUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One cached macro role/assessment compatibility contract,no financial trials."
    )
    parser.add_argument("action", choices=("freeze", "probe"))
    args = parser.parse_args()
    print(json.dumps(freeze() if args.action == "freeze" else probe(), indent=2))
