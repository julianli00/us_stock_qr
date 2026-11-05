from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, market_calendar, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/style-activity-participation-v1.json"
ID = "signed_style_activity_participation_v1"
CORE = ("MTUM", "VLUE", "QUAL", "USMV")
UNIVERSE = (
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
CLOCK = "after_16_15_ET_index_close_before_later_open_execution"
OWN = (
    "config/style-activity-participation-v1.json",
    "src/us_quant/style_activity_participation_v1.py",
    "src/us_quant/style_activity_source_v1.py",
    "tests/test_style_activity_participation_v1.py",
    "STYLE_ACTIVITY_PARTICIPATION_V1.md",
)
FIXTURE_INPUTS = ("tests/conftest.py",)


def validate(policy):
    prefix = "reports/ssd-continuous-research-v1/mixed-mechanism-six-financial-20261105-v1/"
    expected = {
        "schema_version": 1,
        "study_id": "ssd_style_activity_participation_primary_20261105_v1",
        "candidate_id": ID,
        "mechanism_group": "cross_sectional_reported_activity_participation",
        "source_market_freeze": {
            "path": prefix + "financial-implementation-freeze-v1.json",
            "sha256": "f6d9d667554ff1205931f760c56d4365c8e81fd41c17bd485223b773cfcfb06c",
        },
        "source_market_completion": {
            "path": prefix + "completion-v1.json",
            "sha256": "f888b9d4fa1492d2f775480d20cfb82bde183282395333fbace3b57f9c53ced9",
        },
        "activity_admission": {
            "path": "evidence/core_activity_source_readiness_20261029_v1.json",
            "sha256": "09e8dda552865c48398cdb23dac761fab42a3c8ef266eb8de5469e3e6057beee",
        },
        "activity_manifest": {
            "path": "data/ssd-core-activity-source-20261029-v1/manifest.json",
            "sha256": "a96a934ff225f6fd2d16e40a52cb6450ebf55f3f839dfcf18356946165327ba8",
        },
        "prior_completed_configurations": 271,
        "prior_journal_sequence": 1109,
        "prior_journal_head_sha256": (
            "7c0cf8f5a2ab8bd3b2ac8bf2c1c7ae068d51f7fad17a621cf262dd1a2bddb857"
        ),
        "universe": list(UNIVERSE),
        "activity_symbols": list(CORE),
        "cash_asset": "BIL",
        "input_lag_sessions": 1,
        "return_and_activity_observations": 63,
        "required_prior_adjusted_price_levels": 64,
        "minimum_source_position": 64,
        "minimum_positive_activity_balances": 3,
        "selected_styles_when_participating": 3,
        "positive_balance_threshold": 0.0,
        "ranking_decimal_places": 12,
        "invested_target_budget": 0.98,
        "idle_cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "decision_lattice": "genuine_completed_month_ends_after_full_lagged_input_warmup",
        "decision_information_clock": CLOCK,
        "formal_capital_seed_is_new_policy_opinion": False,
        "old_activity_source_RF_imported_or_normalized": False,
        "policy_native_RF": "unchanged_current_frozen_canonical_review_market_RF",
        "source_field_scope_expanded_beyond_original_four": False,
        "new_independent_economic_factor_families": 0,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "parameter_search": False,
        "real_new_indicators_targets_accounts_or_financial_outcomes_computed": False,
        "expert_technically_admitted": False,
        "global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError(
            "Freeze one complete four-style activity law without source or gate rescue."
        )


def balance(closes, reported_closes, reported_volumes) -> float:
    arrays = [np.asarray(p) for p in (closes, reported_closes, reported_volumes)]
    if (
        [p.shape for p in arrays] != [(64,), (63,), (63,)]
        or any(p.dtype.kind not in "fiu" for p in arrays)
        or any(not np.isfinite(p).all() or (p <= 0).any() for p in arrays)
        or not np.array_equal(arrays[2], np.floor(arrays[2]))
    ):
        raise QuantError("Require64actual adjusted closes and63positive native-unit activity rows.")
    close, raw, volume = [p.astype(float, copy=False) for p in arrays]
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="raise"):
            activity = raw * volume
            scaled = activity / activity.max()
        if not np.isfinite(scaled).all() or (scaled <= 0).any():
            raise QuantError("Invalid reported activity cannot receive a cash or uniform fallback.")
        signs = np.sign(np.diff(close))
        value = math.fsum(float(w) * float(s) for w, s in zip(scaled, signs, strict=True))
        value /= math.fsum(float(w) for w in scaled)
    except (FloatingPointError, OverflowError) as error:
        raise QuantError("Undefined signed activity cannot become a safe opinion.") from error
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise QuantError("The fixed direction-weighted activity fraction must remain finite.")
    return value


def allocation(scores: dict) -> tuple[pd.Series, dict]:
    if set(scores) != set(CORE) or any(
        type(v) not in (int, float) or not math.isfinite(v) or not -1 <= v <= 1
        for v in scores.values()
    ):
        raise QuantError("Every admitted style needs a defined balance; missing is not risk-off.")
    positive = [s for s in CORE if scores[s] > 0]
    ranked = sorted(positive, key=lambda s: (-round(scores[s], 12), s))
    selected = ranked[:3] if len(positive) >= 3 else ["BIL"]
    target = pd.Series(0.0, index=UNIVERSE)
    target.loc[selected] = 0.98 / len(selected)
    return target, {
        "activity_balances": scores,
        "ranking_keys": {s: round(scores[s], 12) for s in CORE},
        "strictly_positive_style_count": len(positive),
        "selected_assets": selected,
        "reason": "three_positive_activity_styles"
        if len(positive) >= 3
        else "defined_insufficient_positive_activity_participation",
        "invalid_input_defaulted_to_cash": False,
    }


def checked_market(data: MarketData):
    data.validate()
    if (
        tuple(data.close.columns) != UNIVERSE
        or not data.close.index.equals(sessions(data.close.index[0], data.close.index[-1]))
        or data.close.index[-1] >= market_calendar().last_session
    ):
        raise QuantError("Keep the full admitted fourteen-asset calendar and covered next session.")


def eligible_sessions(data: MarketData) -> pd.DatetimeIndex:
    checked_market(data)
    return pd.DatetimeIndex(
        [day for i, day in enumerate(data.close.index) if i >= 64 and is_month_end(day)]
    )


def _window(data: MarketData, position: int, symbol: str):
    return (
        data.close[symbol].iloc[position - 64 : position],
        data.raw_close[symbol].iloc[position - 63 : position],
        data.volume[symbol].iloc[position - 63 : position],
    )


def decision(data: MarketData, as_of: pd.Timestamp) -> tuple[pd.Series, dict]:
    checked_market(data)
    if (
        not isinstance(as_of, pd.Timestamp)
        or pd.isna(as_of)
        or as_of.tzinfo is not None
        or as_of != as_of.normalize()
        or as_of not in data.close.index
        or not is_month_end(as_of)
    ):
        raise QuantError("Only a genuine covered month end emits a new source policy opinion.")
    position = int(data.close.index.get_loc(as_of))
    if position < 64:
        raise QuantError("A missing complete lagged input window cannot become a BIL opinion.")
    values = {s: balance(*_window(data, position, s)) for s in CORE}
    target, state = allocation(values)
    return target, {
        "decision_session": str(as_of.date()),
        "decision_information_clock": CLOCK,
        "first_prior_adjusted_close": str(data.close.index[position - 64].date()),
        "first_activity_session": str(data.close.index[position - 63].date()),
        "last_input_session": str(data.close.index[position - 1].date()),
        "return_and_activity_observations": 63,
        **state,
        "decision_session_own_or_future_bar_used": False,
        "reported_activity_is_actual_flow_trade_direction_or_market_breadth": False,
        "price_warmup_implies_mature_NET_score": False,
        "counterfactual_historical_availability": True,
        "actual_policy_introduction_or_admission_backdated": False,
        "independent_forward_validation": False,
    }


def build_targets(data: MarketData, policy: dict, before_decision=None, retain=None):
    validate(policy)
    dates = eligible_sessions(data)
    if dates.empty:
        raise QuantError("No complete lagged genuine month end is available; do not fill targets.")
    target = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    audits = []
    for day in dates:
        if before_decision is not None:
            before_decision()
        position = int(data.close.index.get_loc(day))
        if retain is not None:
            retain(
                {
                    "record": "decision_input",
                    "decision_session": str(day.date()),
                    "information_clock": CLOCK,
                    "activity_symbols": list(CORE),
                    "activity_session_dates": data.close.index[position - 63 : position]
                    .strftime("%Y-%m-%d")
                    .tolist(),
                    "inputs_binary64_hex": {
                        s: {
                            key: [float(v).hex() for v in values]
                            for key, values in zip(
                                ("adjusted_closes", "reported_closes", "reported_share_volumes"),
                                _window(data, position, s),
                                strict=True,
                            )
                        }
                        for s in CORE
                    },
                    "decision_session_own_or_future_bar_used": False,
                }
            )
        row, audit = decision(data, day)
        target.loc[day] = row
        audits.append(audit)
        if retain is not None:
            retain(
                {
                    "record": "decision_resolved",
                    "audit": audit,
                    "asset_order": list(UNIVERSE),
                    "complete_target_binary64_hex": [float(v).hex() for v in row],
                }
            )
    return target, audits
