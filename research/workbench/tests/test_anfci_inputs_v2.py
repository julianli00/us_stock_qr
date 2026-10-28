from __future__ import annotations

from zipfile import ZipFile

import numpy as np
import pytest

from us_quant import anfci_inputs_v1 as original
from us_quant import anfci_inputs_v2 as corrected
from us_quant.config import QuantError
from us_quant.storage import file_digest, read_json

DATES = ["2015-08-28", "2015-09-30"]


def archive(tmp_path, missing=""):
    path = tmp_path / "source.zip"
    text = (
        "Series ID: ANFCI\nChicago Fed Adjusted National Financial Conditions Index\n"
        "Federal Reserve Bank of Chicago\nIndex\nWeekly, Ending Friday\nNot Seasonally Adjusted\n"
        "Output Format: Observations by Vintage Date, All Observations\n"
        "Vintage Dates Specified:\n----------\n2015-08-28 2015-09-30\n----------\n"
    )
    with ZipFile(path, "w") as out:
        out.writestr("README.txt", text)
        out.writestr(
            "vintages_starting_2015-08-28.csv",
            "observation_date,ANFCI_20150828,ANFCI_20150930\n"
            f"2015-08-21,-0.12,-0.11\n2015-09-25,{missing},0\n",
        )
    return path


@pytest.mark.parametrize("missing", ["", "."])
def test_native_absence_stays_missing_without_changing_raw_bytes(tmp_path, missing):
    path = archive(tmp_path, missing)
    before = file_digest(path)
    frame, proof = corrected.parse_archive(path, DATES, read_json(original.POLICY))
    assert file_digest(path) == before
    assert np.isnan(frame.iloc[1, 0]) and frame.iloc[0, 0] == -0.12 and frame.iloc[1, 1] == 0
    assert proof["finite_cells_independently_compared"] == 3
    assert proof["finite_source_values_changed"] is False
    if missing == "":
        with pytest.raises(QuantError):
            original.parse_archive(path, DATES, read_json(original.POLICY))


@pytest.mark.parametrize("bad", ["NA", "nan", "inf", "0.5"])
def test_unknown_token_or_future_finite_value_is_not_rescued(tmp_path, bad):
    with pytest.raises(QuantError):
        corrected.parse_archive(archive(tmp_path, bad), DATES, read_json(original.POLICY))
