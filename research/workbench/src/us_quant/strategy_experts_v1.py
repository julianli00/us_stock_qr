from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from us_quant import ssd_research_v1 as base
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.daily_variance_disagreement_v1 import UNIVERSE
from us_quant.storage import digest_json, read_json


@dataclass(frozen=True)
class ExpertIdentity:
    identifier: str
    mechanism_group: str
    introduced_at: str


@dataclass(frozen=True)
class ExpertObservation:
    identity: ExpertIdentity
    as_of: pd.Timestamp
    target: pd.Series | None
    target_known_at: pd.Timestamp | None
    net_returns: pd.Series
    risk_free: pd.Series
    score_mature: bool
    historical_counterfactual: bool = True


class StrategyExpert(Protocol):
    def observe(self, as_of: pd.Timestamp, lookback: int) -> ExpertObservation: ...


class _ArchivedExpert:
    def __init__(
        self,
        identity: ExpertIdentity,
        targets: pd.DataFrame,
        shadow: pd.DataFrame,
        universe: tuple[str, ...],
    ):
        introduced = pd.Timestamp(identity.introduced_at)
        if (
            not identity.identifier
            or not identity.mechanism_group
            or introduced.tzinfo is None
            or pd.isna(introduced)
            or not isinstance(targets.index, pd.DatetimeIndex)
            or targets.empty
            or not targets.index.is_unique
            or not targets.index.is_monotonic_increasing
            or targets.index.tz is not None
            or not targets.index.equals(sessions(targets.index[0], targets.index[-1]))
            or not targets.columns.is_unique
            or not set(targets.columns).issubset(universe)
            or len(set(universe)) != len(universe)
            or (targets.isna().any(axis=1) & ~targets.isna().all(axis=1)).any()
        ):
            raise QuantError(
                "An archived expert needs a real clock and complete admitted asset lattice."
            )
        active = targets.dropna(how="all")
        if (
            active.empty
            or not np.isfinite(active.to_numpy()).all()
            or (active < 0).any().any()
            or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        ):
            raise QuantError("Expert recommendations must be complete long-only funded portfolios.")
        needed = ("equity", "return", "risk_free", "cash", "gross_exposure", "cost", "orders")
        if (
            not isinstance(shadow.index, pd.DatetimeIndex)
            or shadow.empty
            or not shadow.index.is_unique
            or not shadow.index.is_monotonic_increasing
            or shadow.index.tz is not None
            or not shadow.index.equals(sessions(shadow.index[0], shadow.index[-1]))
            or not set(needed).issubset(shadow.columns)
            or not np.isfinite(shadow.loc[:, list(needed)].to_numpy()).all()
            or (shadow["equity"] <= 0).any()
            or (shadow["return"] <= -1).any()
            or (shadow["cash"] < 0).any()
            or (shadow["gross_exposure"] < 0).any()
            or (shadow["gross_exposure"] > 1).any()
            or (shadow["cost"] < 0).any()
            or (shadow["orders"] < 0).any()
            or not shadow.index.isin(targets.index).all()
        ):
            raise QuantError(
                "Shadow observations must be complete, net, cash-funded research records."
            )
        prior_nav = shadow["equity"].shift(1)
        prior_nav.iloc[0] = 10000.0
        if not np.allclose(shadow["equity"] / prior_nav - 1, shadow["return"], rtol=0, atol=1e-12):
            raise QuantError(
                "Shadow returns must reconcile to their own reference capital, not main capital."
            )
        self.identity = identity
        self._targets = targets.copy()
        self._active = active.copy()
        self._shadow = shadow.copy()
        self._universe = universe

    def _known_target(self, as_of):
        raise NotImplementedError

    def observe(self, as_of: pd.Timestamp, lookback: int) -> ExpertObservation:
        day = pd.Timestamp(as_of)
        if day not in self._targets.index or type(lookback) is not int or lookback < 2:
            raise QuantError(
                "Observe a covered actual session with an explicit positive scoring window."
            )
        target, known_at = self._known_target(day)
        observed = self._shadow.loc[:day].tail(lookback)
        if target is not None:
            # Assets outside this complete expert's admitted universe are structural zero exposures.
            target = target.reindex(self._universe, fill_value=0.0).copy()
        return ExpertObservation(
            self.identity,
            day,
            target,
            known_at,
            observed["return"].copy(),
            observed["risk_free"].copy(),
            len(observed) == lookback,
        )


class ArchivedDenseExpert(_ArchivedExpert):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._active.index.equals(self._targets.loc[self._active.index[0] :].index):
            raise QuantError(
                "A dense expert cannot carry across a missing post-warmup recommendation."
            )

    def _known_target(self, as_of):
        if as_of < self._active.index[0]:
            return None, None
        return self._targets.loc[as_of].copy(), as_of


class ArchivedMonthlyExpert(_ArchivedExpert):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        expected = pd.DatetimeIndex(
            [day for day in self._targets.loc[self._active.index[0] :].index if is_month_end(day)]
        )
        if not self._active.index.equals(expected):
            raise QuantError("A monthly expert needs every genuine month end after its warmup.")

    def _known_target(self, as_of):
        known = self._active.loc[:as_of]
        if known.empty:
            return None, None
        return known.iloc[-1].copy(), known.index[-1]


def _frame(reference: dict) -> pd.DataFrame:
    return pd.read_csv(
        base.checked_reference(reference),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )


def load_expert(record: dict, universe: tuple[str, ...]) -> tuple[StrategyExpert, dict]:
    if universe != (*UNIVERSE, "IJR", "PKW"):
        raise QuantError(
            "This artifact Adapter supports only the existing admitted fourteen-asset market."
        )
    packet = read_json(base.checked_reference(record["registration"]))
    source_results = read_json(base.checked_reference(record["results"]))
    entries = [r for r in packet["entries"] if r["candidate_id"] == record["id"]]
    reviews = [r for r in source_results["reviews"] if r["candidate_id"] == record["id"]]
    if len(entries) != 1 or len(reviews) != 1:
        raise QuantError("An expert requires its actual unique registration and completed review.")
    entry, review = entries[0], reviews[0]
    frozen = entry["specification"]["frozen"]
    policy = read_json(base.checked_reference(frozen["policy"]))
    if (
        source_results["registration_packet"] != record["registration"]
        or source_results["protocol"]
        != {"path": "config/dual-benchmark-v1.json", "sha256": base.PROTOCOL_SHA256}
        or entry["specification"]["candidate"] not in policy["candidates"]
        or review["status"] not in {"rejected", "historical_point_gates_passed"}
        or review["all_four_paths_same_candidate_and_frozen_parameters"] is not True
        or packet["candidate_accounts_or_metrics_computed"] is not False
        or source_results["order_authority"] is not False
        or set(record["completed_paths"]) != base.PATH_KEYS
        or entry["registered_at"] != record["introduced_at"]
        or entry["registered_event_sha256"] != record["registered_event_sha256"]
    ):
        raise QuantError(
            "Invalid source identity or technical completeness cannot enter an expert pool."
        )
    base.check_frozen_files(frozen)
    for ref in frozen.get("prior_evidence", []) + frozen.get("validation_evidence", []):
        base.checked_reference(ref)
    proof = read_json(base.checked_reference(entry["target_proof"]["proof"]))
    if (
        proof["candidate_accounts_or_metrics_computed"] is not False
        or proof["original_source_signal_recomputed"] is not False
        or not pd.Timestamp(entry["registered_at"])
        <= pd.Timestamp(proof["checked_at"])
        <= pd.Timestamp(entry["target_proof_at"])
        <= pd.Timestamp(packet["created_at"])
    ):
        raise QuantError(
            "Expert target proof must follow its real registration without rewritten clocks."
        )
    saved = {}
    by_key = {p["path_key"]: p for p in review["paths"]}
    if set(by_key) != base.PATH_KEYS:
        raise QuantError("A complete expert must retain every original funded evidence path.")
    for key, reference in record["completed_paths"].items():
        complete = base.verify_path(reference, frozen, record["id"], key)
        execution = complete["execution"]
        if (
            complete["metrics"] != by_key[key]["metrics"]
            or execution != by_key[key]["execution"]
            or execution["minimum_cash_usd"] < 0
            or not 0 <= execution["maximum_gross_exposure"] <= 1
            or complete["independent_bt_max_equity_difference_usd"] > 1e-6
            or complete["broker_orders"] != 0
        ):
            raise QuantError(
                "Technical funding/accounting validity is required independently of performance."
            )
        saved[key] = complete
    shadow_refs = [r for r in saved["10y-base"]["artifacts"] if r["path"].endswith("/account.csv")]
    if len(shadow_refs) != 1:
        raise QuantError(
            "Use the existing complete ten-year base net shadow book, not averaged metrics."
        )
    shadow = _frame(shadow_refs[0])
    if (
        len(shadow) != 2512
        or str(shadow.index[0].date()) != "2016-10-06"
        or str(shadow.index[-1].date()) != "2026-10-05"
    ):
        raise QuantError("The reference shadow book cannot silently lose formal sessions.")
    targets = _frame(entry["target_proof"]["targets"])
    identity = ExpertIdentity(record["id"], record["mechanism_group"], record["introduced_at"])
    adapters = {"dense": ArchivedDenseExpert, "monthly": ArchivedMonthlyExpert}
    if record["adapter"] not in adapters:
        raise QuantError("Unknown expert recommendation cadence.")
    expert = adapters[record["adapter"]](identity, targets, shadow, universe)
    return expert, {
        "id": record["id"],
        "mechanism_group": record["mechanism_group"],
        "adapter": record["adapter"],
        "introduced_at": record["introduced_at"],
        "first_historical_target": str(targets.dropna(how="all").index[0].date()),
        "historical_shadow_start": "2016-10-06",
        "historical_shadow_end": "2026-10-05",
        "historical_shadow_observations": 2512,
        "target_reference": entry["target_proof"]["targets"],
        "shadow_reference": shadow_refs[0],
        "source_frozen_specification_sha256": digest_json(frozen),
        "standalone_financial_verdict_preserved": review["status"],
        "financial_failure_is_not_admission_failure": True,
        "source_technical_and_funding_checks_passed": True,
        "four_completed_paths_checked": 4,
        "counterfactual_historical_availability": True,
        "independent_forward_validation": False,
    }
