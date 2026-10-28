from __future__ import annotations

import csv
import io

import numpy as np
import pandas as pd

from us_quant import anfci_inputs_v1 as original
from us_quant.config import QuantError
from us_quant.storage import read_json

base, ROOT = original.base, original.ROOT
POLICY = ROOT / "config/anfci-native-csv-v2.json"
DATA = ROOT / "data/ssd-anfci-vintages-20261028-v2"
REPORTS = base.REPORTS / "anfci-vintage-admission-20261028-v2"
PUBLIC = ROOT / "evidence/anfci_input_readiness_20261028_v2.json"
MISSING = (".", "")


def settings() -> tuple[dict, dict]:
    correction = read_json(POLICY)
    expected = {
        "schema_version": 1,
        "source_id": "ssd_anfci_vintages_20261028_v2",
        "implementation_revision": 2,
        "native_missing_tokens": list(MISSING),
        "data_rule_parameters_changed": False,
        "original_contract_or_data_modified": False,
        "repeat_first_archive_request_allowed": False,
        "first_archive_predates_v2_correction": True,
        "remaining_requests_require_v2_freeze": True,
        "ANFCI_strategy_outcomes_computed": False,
        "new_strategy_evaluations": 0,
        "new_factor_definitions": 0,
        "order_authority": False,
    }
    if any(
        key not in correction
        or correction[key] != value
        or type(correction[key]) is not type(value)
        for key, value in expected.items()
    ):
        raise QuantError("Only the explicitly versioned native empty-cell correction is allowed.")
    for name in (
        "original_contract",
        "original_failure",
        "diagnostic",
        "retained_attempt",
        "retained_request",
        "retained_response",
    ):
        base.checked_reference(correction[name])
    frozen = read_json(base.checked_reference(correction["original_contract"]))
    for ref in frozen["inputs"]:
        base.checked_reference(ref)
    matches = [ref for ref in frozen["inputs"] if ref["path"] == "config/anfci-inputs-v1.json"]
    if len(matches) != 1:
        raise QuantError("The original source policy must remain exactly bound.")
    policy = read_json(base.checked_reference(matches[0]))
    original.validate(policy)
    return policy, correction


def parse_archive(path, dates, policy):
    readme, content = original.archive_contents(path, dates, policy)
    frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    names = ["observation_date", *[f"ANFCI_{day.replace('-', '')}" for day in dates]]
    if list(frame.columns) != names:
        raise QuantError("Native CSV columns must retain every original requested vintage.")
    try:
        index = pd.DatetimeIndex(
            pd.to_datetime(frame.pop("observation_date"), format="%Y-%m-%d", errors="raise")
        )
    except ValueError as error:
        raise QuantError("Native CSV observation dates are malformed.") from error
    if (
        index.empty
        or index.has_duplicates
        or not index.is_monotonic_increasing
        or not (index.weekday == 4).all()
        or index[0] < pd.Timestamp(policy["observation_start"])
        or index[-1] > pd.Timestamp(policy["observation_end"])
    ):
        raise QuantError("Native CSV must retain unique chronological Friday dates in range.")
    result = pd.DataFrame(index=index)
    blank, dot = 0, 0
    for day, name in zip(dates, names[1:], strict=True):
        blank += int(frame[name].eq("").sum())
        dot += int(frame[name].eq(".").sum())
        missing = frame[name].isin(MISSING).to_numpy()
        try:
            values = pd.to_numeric(frame[name].mask(missing), errors="raise").to_numpy(dtype=float)
        except ValueError as error:
            raise QuantError(
                "An unknown nonnumeric token is not an allowed source absence."
            ) from error
        if (
            not np.array_equal(np.isnan(values), missing)
            or np.isinf(values).any()
            or missing.all()
            or (index[~missing] > pd.Timestamp(day)).any()
        ):
            raise QuantError(
                "Invalid, wholly absent or future finite observations remain prohibited."
            )
        result[day] = values
    independent = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    if independent[0] != names or len(independent) - 1 != len(result):
        raise QuantError("Independent native CSV shape differs.")
    maximum, compared = 0.0, 0
    for position, row in enumerate(independent[1:]):
        if len(row) != len(names) or row[0] != str(index[position].date()):
            raise QuantError("Independent native CSV date/column shape differs.")
        for column, cell in enumerate(row[1:]):
            actual = result.iloc[position, column]
            if cell in MISSING:
                if not pd.isna(actual):
                    raise QuantError("A native absent cell was filled or reclassified as a value.")
            else:
                value = float(cell)
                if not np.isfinite(value) or not np.isfinite(actual):
                    raise QuantError("Independent native numeric parsing failed.")
                maximum = max(maximum, abs(value - float(actual)))
                compared += 1
    if maximum > 1e-12:
        raise QuantError("Independent finite values differ from the unchanged raw observations.")
    return result, {
        "observation_rows": len(result),
        "vintage_columns": len(dates),
        "native_empty_missing_cells": blank,
        "native_dot_missing_cells": dot,
        "finite_cells_independently_compared": compared,
        "maximum_independent_numeric_difference": maximum,
        "README_vintage_metadata_verified": True,
        "README_text_length": len(readme),
        "missing_cells_preserved_not_filled": True,
        "finite_source_values_changed": False,
    }
