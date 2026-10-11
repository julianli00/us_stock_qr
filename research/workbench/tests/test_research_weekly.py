from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
from test_research_program import candidate_spec, synthetic_market

from us_quant import research_weekly as weekly
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.research_program import ResearchProgram
from us_quant.storage import digest_json, file_digest, read_json, write_json
from us_quant.strategy import buy_and_hold_signals

ROOT = Path(__file__).parents[1]
NOW = pd.Timestamp("2026-10-17T01:01:00Z")
TRAILER = "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"


def command(repo, *args):
    return (
        subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .strip()
    )


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    root = repo / "research/workbench"
    root.mkdir(parents=True)
    command(repo, "init", "--quiet", "--initial-branch=research")
    command(repo, "config", "user.name", "Synthetic Research")
    command(repo, "config", "user.email", "synthetic@example.invalid")
    remote = tmp_path / "remote.git"
    command(repo, "init", "--bare", "--quiet", str(remote))
    command(repo, "remote", "add", "origin", str(remote))
    sources = (
        "src/us_quant/research_weekly.py",
        "src/us_quant/research_program.py",
        "src/us_quant/strategy_replay.py",
        "src/us_quant/candidate_receipts.py",
        "src/us_quant/review_receipts.py",
        "src/us_quant/cash_funded_accounting_v2.py",
        "src/us_quant/bt_audit.py",
        "config/research-program.json",
        "config/factor-family-expansion.json",
        "config/six-factor-strategy.json",
        "config/inflation-factor-guard.json",
        "config/claims-factor-guard.json",
    )
    for source in sources:
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, path)
    policy = read_json(root / "config/research-program.json")
    proposals = read_json(root / "config/factor-family-expansion.json")
    definitions = {
        factor["id"]: factor for factor in [*policy["seed_factors"], *proposals["factors"]]
    }
    data, market = synthetic_market(root, winning=False)
    proof = root / "proof.json"
    write_json(proof, {"synthetic_scope_only": True})
    first_readiness = None
    for candidate, group, generator in weekly.AGENDA:
        source_policy = read_json(
            root / weekly.registered_generator.__globals__["GENERATORS"][generator]
        )
        source_candidate = next(
            item for item in source_policy["candidates"] if item["id"] == candidate
        )
        factor_ids = list(definitions) if group == "six-factor" else list(definitions)[:4]
        specification = candidate_spec(root, identifier=candidate, winning=False)
        specification.update(
            {
                "data_scope": "factor_etf_portfolio",
                "factor_ids": factor_ids,
                "market": market,
                "configuration": source_candidate,
            }
        )
        readiness = {
            "schema_version": 1,
            "checked_at": NOW.isoformat(),
            "data_scope": "factor_etf_portfolio",
            "synthetic_expected_factor_ids": factor_ids,
            "capabilities": {
                key: {
                    "verified": True,
                    "evidence_path": proof.name,
                    "evidence_sha256": file_digest(proof),
                }
                for key in (
                    "factor_mandates",
                    "actual_fund_history",
                    "post_inception_and_actions",
                    "unleveraged_fund_identity",
                )
            },
        }
        directory = root / f"data/{group}-prepared-20261017"
        write_json(directory / f"{candidate}-spec.json", specification)
        if (directory / "readiness.json").exists():
            assert read_json(directory / "readiness.json") == readiness
        else:
            write_json(directory / "readiness.json", readiness)
        if first_readiness is None:
            first_readiness = readiness
    monkeypatch.setattr(
        "us_quant.research_program.timestamp",
        lambda now=None: (NOW.to_pydatetime() if now is None else now),
    )
    monkeypatch.setattr("us_quant.candidate_receipts.timestamp", lambda: NOW.to_pydatetime())
    monkeypatch.setattr("us_quant.review_receipts.timestamp", lambda: NOW.to_pydatetime())
    monkeypatch.setattr("us_quant.research_program.verified_etf_market", lambda *args: data)
    monkeypatch.setattr(
        "us_quant.research_program.verified_etf_definitions",
        lambda readiness, *args: {
            name: definitions[name] for name in readiness["synthetic_expected_factor_ids"]
        },
    )
    monkeypatch.setattr(
        "us_quant.research_program.registered_targets",
        lambda spec, data, start, *args: (buy_and_hold_signals(data.close, "TEST", start) * 0.98),
    )
    monkeypatch.setattr("us_quant.candidate_receipts.check_eligibility", lambda *args: None)
    monkeypatch.setattr(
        "us_quant.claims_factor_guard.check_registration_eligibility", lambda *args: None
    )
    monkeypatch.setattr(
        "us_quant.inflation_factor_guard.check_registration_eligibility", lambda *args: None
    )
    generator_by_id = {row[0]: row[2] for row in weekly.AGENDA}
    monkeypatch.setattr(
        weekly,
        "registered_generator",
        lambda spec, root: (
            generator_by_id[spec["id"]],
            read_json(
                root
                / {
                    "six_factor_strategy": "config/six-factor-strategy.json",
                    "inflation_factor_guard": "config/inflation-factor-guard.json",
                    "claims_factor_guard": "config/claims-factor-guard.json",
                }[generator_by_id[spec["id"]]]
            ),
        ),
    )
    program = ResearchProgram(
        root / "runtime/research-program.sqlite3",
        policy,
        root=root,
        create=True,
    )
    try:
        program.cycle(proposals, first_readiness, now=NOW.to_pydatetime())
    finally:
        program.close()
    return repo, root


def state(root):
    program = ResearchProgram(
        root / "runtime/research-program.sqlite3",
        read_json(root / "config/research-program.json"),
        root=root,
    )
    try:
        return program.status()
    finally:
        program.close()


def run(root, action="register", **kwargs):
    return weekly.run(action, "operator", root=root, clock=lambda: NOW, **kwargs)


def publish(repo, root, *, push=True):
    packet = read_json(root / weekly.PACKET)
    entries = packet["candidates"]
    prefix = root.relative_to(repo).as_posix()
    paths = sorted(
        {
            "config/publication_allowlist.json",
            f"{prefix}/{weekly.PACKET}",
            *(f"{prefix}/{name}" for name in weekly.required_public_files(root, entries)),
        }
    )
    write_json(repo / "config/publication_allowlist.json", {"schema_version": 1, "files": paths})
    command(repo, "add", "--", *paths)
    command(repo, "commit", "--quiet", "-m", "Synthetic preregistration", "-m", TRAILER)
    commit = command(repo, "rev-parse", "HEAD")
    if push:
        command(repo, "push", "--quiet", "--set-upstream", "origin", "research")
    return commit


def test_actual_early_clock_records_explicit_failure_without_admission_or_evaluation(tmp_path):
    earlier = pd.Timestamp("2026-10-11T08:40:00Z")
    with pytest.raises(QuantError, match="not eligible before"):
        weekly.run("register", "session_automation", root=tmp_path, clock=lambda: earlier)
    receipt = read_json(next((tmp_path / weekly.DIRECTORY / "runs").glob("*/result.json")))
    assert receipt["status"] == "failed" and receipt["steps"] == []
    assert not (tmp_path / weekly.PACKET).exists()
    assert not receipt["native_schedule_delivery_independently_verified"]
    assert not receipt["investment_objective_verified"] and not receipt["order_authority"]


def test_registration_packet_contains_four_exact_committed_specs_and_no_performance(prepared):
    _, root = prepared
    before = state(root)
    result = run(root)
    packet = read_json(root / weekly.PACKET)
    after = state(root)
    assert result["status"] == "registered_waiting_for_publication"
    assert result["new_candidate_registration_count"] == 4
    assert result["new_historical_strategy_evaluations"] == 0
    assert after["total_evaluated_configurations"] == before["total_evaluated_configurations"]
    assert [entry["spec"]["id"] for entry in packet["candidates"]] == [
        row[0] for row in weekly.AGENDA
    ]
    assert not packet["candidate_portfolio_returns_computed_when_packet_created"]
    digest = file_digest(root / weekly.PACKET)
    repeated = run(root)
    assert repeated["new_candidate_registration_count"] == 0
    assert file_digest(root / weekly.PACKET) == digest
    assert state(root) == after


def test_interrupted_registration_keeps_committed_candidate_and_restores_same_original_event(
    prepared, monkeypatch
):
    _, root = prepared
    original = weekly.register_or_recover
    calls = []

    def interrupted(spec, readiness, receipt, **kwargs):
        calls.append(spec.name)
        if len(calls) == 2:

            def fail(*args):
                raise OSError("synthetic post-commit receipt error")

            return original(spec, readiness, receipt, receipt_writer=fail, **kwargs)
        return original(spec, readiness, receipt, **kwargs)

    monkeypatch.setattr(weekly, "register_or_recover", interrupted)
    with pytest.raises(OSError, match="post-commit"):
        run(root)
    before = state(root)
    assert len(before["registered_candidates"]) == 2
    assert before["candidate_reviews"] == []
    monkeypatch.setattr(weekly, "register_or_recover", original)
    result = run(root)
    assert result["new_candidate_registration_count"] == 2
    assert len(state(root)["registered_candidates"]) == 4
    packet = read_json(root / weekly.PACKET)
    previous_by_id = {item["id"]: item for item in before["registered_candidates"]}
    assert packet["candidates"][0]["registration"]["spec_sha256"] == digest_json(
        previous_by_id[weekly.AGENDA[0][0]],
    )
    assert [step["result"]["action"] for step in result["steps"]][:2] == [
        "already_recorded",
        "recovered_registration",
    ]


def test_receipt_failure_stops_later_steps_and_retains_actual_failure_record(prepared, monkeypatch):
    _, root = prepared
    monkeypatch.setattr(
        weekly,
        "register_or_recover",
        lambda *args, **kwargs: (
            {"candidate_id": "unrelated", "action": "registered", "order_authority": True}
        ),
    )
    with pytest.raises(QuantError, match="unsupported candidate scope"):
        run(root)
    receipt = read_json(next((root / weekly.DIRECTORY / "runs").glob("*/result.json")))
    assert receipt["status"] == "failed" and len(receipt["steps"]) == 1
    assert receipt["steps"][0]["result"]["candidate_id"] == "unrelated"
    assert not (root / weekly.PACKET).exists()
    assert state(root)["registered_candidates"] == []


def test_missing_preparation_is_not_created_or_filled_by_the_registration_queue(prepared):
    _, root = prepared
    missing = root / "data/claims-guard-prepared-20261017/readiness.json"
    missing.unlink()
    before = state(root)
    with pytest.raises(QuantError, match="missing"):
        run(root)
    assert state(root) == before
    assert not (root / weekly.PACKET).exists()


def test_no_actual_factor_cycle_cannot_be_replaced_by_a_different_week(prepared):
    _, root = prepared
    original = weekly.WEEK
    weekly.WEEK = "2026-W43"
    try:
        with pytest.raises(QuantError, match="actual IJR/PKW"):
            run(root)
    finally:
        weekly.WEEK = original
    assert state(root)["registered_candidates"] == []


def test_local_commit_is_not_remote_publication_and_never_starts_evaluation(prepared):
    repo, root = prepared
    run(root)
    commit = publish(repo, root, push=False)
    before = state(root)
    with pytest.raises(QuantError, match="configured remote"):
        run(root, "evaluate", published_commit=commit)
    assert state(root) == before
    assert not (root / "reports/six-factor-20261017").exists()


def test_remote_publication_allows_original_evaluation_and_repeated_recovery_is_not_a_new_trial(
    prepared,
):
    repo, root = prepared
    run(root)
    commit = publish(repo, root)
    result = run(root, "evaluate", published_commit=commit)
    after = state(root)
    assert result["status"] == "all_original_outcomes_recorded"
    assert result["new_historical_strategy_evaluations"] == 4
    assert len(after["candidate_reviews"]) == 4
    assert all(review["status"] == "rejected_historical" for review in after["candidate_reviews"])
    assert not result["investment_objective_verified"]
    assert not result["publication_fence"]["verification_is_independent_first_publication_time"]
    receipt_path = root / weekly.DIRECTORY / "publication.json"
    digest = file_digest(receipt_path)
    repeated = run(root, "evaluate", published_commit=commit)
    assert repeated["new_historical_strategy_evaluations"] == 0
    assert state(root) == after and file_digest(receipt_path) == digest
    assert all(step["result"]["action"] == "already_recorded" for step in repeated["steps"][1:])


@pytest.mark.parametrize("problem", ["code", "packet", "allowlist"])
def test_changed_publication_bytes_or_boundary_stop_before_any_account(prepared, problem):
    repo, root = prepared
    run(root)
    commit = publish(repo, root)
    before = state(root)
    if problem == "code":
        (root / "src/us_quant/research_weekly.py").write_text("changed frozen queue")
    elif problem == "packet":
        packet = read_json(root / weekly.PACKET)
        packet["candidates"][0]["spec"]["configuration"] = {"unrequested": True}
        write_json(root / weekly.PACKET, packet)
    else:
        allowlist = read_json(repo / "config/publication_allowlist.json")
        allowlist["files"].remove("research/workbench/src/us_quant/research_weekly.py")
        write_json(repo / "config/publication_allowlist.json", allowlist)
    with pytest.raises(QuantError):
        run(root, "evaluate", published_commit=commit)
    assert state(root) == before
    assert not (root / "reports/six-factor-20261017").exists()


def test_later_unrelated_approved_publication_does_not_rewrite_the_original_registration_fence(
    prepared,
):
    repo, root = prepared
    run(root)
    commit = publish(repo, root)
    run(root, "evaluate", published_commit=commit)
    original = file_digest(root / weekly.DIRECTORY / "publication.json")
    write_json(repo / "unrelated.json", {"synthetic_unrelated_change": True})
    allowlist = read_json(repo / "config/publication_allowlist.json")
    allowlist["files"].append("unrelated.json")
    write_json(repo / "config/publication_allowlist.json", allowlist)
    command(repo, "add", "--", "unrelated.json", "config/publication_allowlist.json")
    command(repo, "commit", "--quiet", "-m", "Synthetic later metadata", "-m", TRAILER)
    command(repo, "push", "--quiet", "origin", "research")
    before = state(root)
    result = run(root, "evaluate", published_commit=commit)
    assert result["new_historical_strategy_evaluations"] == 0
    assert state(root) == before
    assert file_digest(root / weekly.DIRECTORY / "publication.json") == original


def test_uncommitted_local_registration_cannot_be_called_remote_publication(prepared):
    repo, root = prepared
    run(root)
    commit = publish(repo, root)
    path = root / weekly.PACKET
    path.write_text(json.dumps(read_json(path), separators=(",", ":")))
    before = state(root)
    with pytest.raises(QuantError, match="differs from frozen local bytes"):
        run(root, "evaluate", published_commit=commit)
    assert state(root) == before


def test_partial_existing_evaluation_directory_is_preserved_and_blocks_the_whole_queue(prepared):
    repo, root = prepared
    run(root)
    commit = publish(repo, root)
    path = (
        root / "reports/claims-guard-20261017/semiconductor_dollar_claims_confirmation/partial.csv"
    )
    path.parent.mkdir(parents=True)
    path.write_text("synthetic incomplete output")
    before, digest = state(root), file_digest(path)
    with pytest.raises(QuantError, match="missing"):
        run(root, "evaluate", published_commit=commit)
    assert state(root) == before and file_digest(path) == digest
    assert not (root / "reports/six-factor-20261017").exists()


def test_performance_before_publication_cannot_be_relabelled_as_preregistered(prepared):
    repo, root = prepared
    run(root)
    commit = publish(repo, root)
    path = root / "reports/six-factor-20261017/six_factor_equal_family/bundle.json"
    write_json(path, {"completed_at": "2026-10-17T01:00:00Z"})
    before = state(root)
    with pytest.raises(QuantError, match="Pre-publication evaluation"):
        run(root, "evaluate", published_commit=commit)
    assert state(root) == before


def test_two_concurrent_registration_calls_keep_one_identical_packet_and_four_events(prepared):
    _, root = prepared
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: run(root), range(2)))
    assert sum(result["new_candidate_registration_count"] for result in results) == 4
    assert results[0]["registration_packet"] == results[1]["registration_packet"]
    assert len(state(root)["registered_candidates"]) == 4


@pytest.mark.parametrize("commit", [None, "main", "1" * 8, "bad;git push", "f" * 40])
def test_exact_commit_and_existing_remote_proof_are_required(prepared, commit):
    repo, root = prepared
    run(root)
    publish(repo, root)
    before = state(root)
    with pytest.raises(QuantError):
        run(root, "evaluate", published_commit=commit)
    assert state(root) == before


def test_a_missing_registration_packet_writer_cannot_report_success(prepared, monkeypatch):
    _, root = prepared
    monkeypatch.setattr(weekly, "write_new_receipt", lambda *args: None)
    with pytest.raises(QuantError, match="missing"):
        run(root)
    assert len(state(root)["registered_candidates"]) == 4
    assert not (root / weekly.PACKET).exists()
    monkeypatch.setattr(weekly, "write_new_receipt", write_new_receipt)
    result = run(root)
    assert result["new_candidate_registration_count"] == 0


def test_complete_bundle_before_review_commit_recovers_without_regenerating_curves(
    prepared,
    monkeypatch,
):
    repo, root = prepared
    run(root)
    commit = publish(repo, root)
    original = ResearchProgram.review

    def fail_before_commit(*args, **kwargs):
        raise OSError("synthetic interruption before review")

    monkeypatch.setattr(ResearchProgram, "review", fail_before_commit)
    with pytest.raises(OSError, match="before review"):
        run(root, "evaluate", published_commit=commit)
    bundle = root / "reports/six-factor-20261017/six_factor_equal_family/bundle.json"
    digest = file_digest(bundle)
    assert state(root)["candidate_reviews"] == []
    monkeypatch.setattr(ResearchProgram, "review", original)
    result = run(root, "evaluate", published_commit=commit)
    assert result["new_historical_strategy_evaluations"] == 4
    assert result["steps"][1]["result"]["action"] == "review_recorded"
    assert file_digest(bundle) == digest


def test_committed_review_before_output_return_recovers_receipt_without_recounting(
    prepared,
    monkeypatch,
):
    repo, root = prepared
    run(root)
    commit = publish(repo, root)
    original = ResearchProgram.evaluate_registered

    def commit_then_fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise OSError("synthetic interruption after evaluation commit")

    monkeypatch.setattr(ResearchProgram, "evaluate_registered", commit_then_fail)
    with pytest.raises(OSError, match="after evaluation commit"):
        run(root, "evaluate", published_commit=commit)
    before = state(root)
    assert len(before["candidate_reviews"]) == 1
    monkeypatch.setattr(ResearchProgram, "evaluate_registered", original)
    result = run(root, "evaluate", published_commit=commit)
    assert result["new_historical_strategy_evaluations"] == 3
    assert result["steps"][1]["result"]["action"] == "recovered_review"
    original_review = before["candidate_reviews"][0]
    after = state(root)
    assert (
        next(
            review
            for review in after["candidate_reviews"]
            if review["candidate_id"] == original_review["candidate_id"]
        )
        == original_review
    )


def test_missing_publication_receipt_writer_never_starts_a_portfolio(prepared, monkeypatch):
    repo, root = prepared
    run(root)
    commit = publish(repo, root)
    before = state(root)
    monkeypatch.setattr(weekly, "write_new_receipt", lambda *args: None)
    with pytest.raises(QuantError, match="missing"):
        run(root, "evaluate", published_commit=commit)
    assert state(root) == before
    assert not (root / "reports/six-factor-20261017").exists()


def test_post_publication_receipt_write_interruption_retains_the_same_original_fence(
    prepared,
    monkeypatch,
):
    repo, root = prepared
    run(root)
    commit = publish(repo, root)

    def write_then_fail(path, receipt):
        write_new_receipt(path, receipt)
        raise OSError("synthetic after publication receipt write")

    monkeypatch.setattr(weekly, "write_new_receipt", write_then_fail)
    with pytest.raises(OSError, match="after publication"):
        run(root, "evaluate", published_commit=commit)
    receipt = root / weekly.DIRECTORY / "publication.json"
    digest = file_digest(receipt)
    assert state(root)["candidate_reviews"] == []
    monkeypatch.setattr(weekly, "write_new_receipt", write_new_receipt)
    result = run(root, "evaluate", published_commit=commit)
    assert result["new_historical_strategy_evaluations"] == 4
    assert file_digest(receipt) == digest


@pytest.mark.parametrize("problem", ["future_time", "different_commit", "partial"])
def test_changed_publication_fence_is_preserved_and_never_replaced(prepared, problem):
    repo, root = prepared
    run(root)
    commit = publish(repo, root)
    run(root, "evaluate", published_commit=commit)
    receipt = root / weekly.DIRECTORY / "publication.json"
    if problem == "partial":
        receipt.write_text("{partial")
    else:
        content = read_json(receipt)
        if problem == "future_time":
            content["verified_on_remote_at"] = (NOW + pd.Timedelta(days=1)).isoformat()
        else:
            content["published_commit"] = "0" * 40
        write_json(receipt, content)
    before, digest = state(root), file_digest(receipt)
    with pytest.raises(QuantError):
        run(root, "evaluate", published_commit=commit)
    assert state(root) == before and file_digest(receipt) == digest


def test_a_local_descendant_commit_not_on_the_actual_remote_is_not_published(prepared):
    repo, root = prepared
    run(root)
    publish(repo, root)
    write_json(repo / "unpublished.json", {"synthetic_local_only": True})
    command(repo, "add", "--", "unpublished.json")
    command(repo, "commit", "--quiet", "-m", "Synthetic unpublished change", "-m", TRAILER)
    commit = command(repo, "rev-parse", "HEAD")
    before = state(root)
    with pytest.raises(QuantError, match="merge-base"):
        run(root, "evaluate", published_commit=commit)
    assert state(root) == before
    assert not (root / "reports/six-factor-20261017").exists()


def test_changed_factor_proposals_cannot_substitute_another_weekly_definition_cycle(prepared):
    _, root = prepared
    proposals = read_json(root / "config/factor-family-expansion.json")
    proposals["factors"][0]["limitations"] = "unrequested replacement"
    write_json(root / "config/factor-family-expansion.json", proposals)
    before = state(root)
    with pytest.raises(QuantError, match="substitute another cycle"):
        run(root)
    assert state(root) == before


def test_backwards_weekly_stage_clock_stops_without_candidate_admission(prepared):
    _, root = prepared
    times = iter((NOW, NOW - pd.Timedelta(seconds=1)))
    before = state(root)
    with pytest.raises(QuantError, match="timestamps cannot move backwards"):
        weekly.run("register", "operator", root=root, clock=lambda: next(times))
    assert state(root) == before
