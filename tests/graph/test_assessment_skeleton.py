from __future__ import annotations

"""Phase 0 exit-criteria test: a run can be created, checkpointed, interrupted,
resumed, and completed with zero LLM/network calls.

See docs/LangGraph Implementation Specification.md, Phase 0 exit condition.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from langgraph.types import Command  # noqa: E402

from manuscript_system.graphs.assessment import build_assessment_graph, initial_state  # noqa: E402
from manuscript_system.persistence.database import connect  # noqa: E402
from manuscript_system.persistence.repositories import Repository  # noqa: E402


def make_repo(tmp_path: Path) -> Repository:
    conn = connect(str(tmp_path / "test.sqlite"))
    return Repository(conn)


def test_run_can_be_created_checkpointed_interrupted_and_resumed(tmp_path):
    repo = make_repo(tmp_path)
    graph = build_assessment_graph(repo, str(tmp_path / "checkpoints.sqlite"))

    thread_id = "thread-test-0001"
    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(
        source_path=str(tmp_path), project_id="PROJECT-TEST", run_id="RUN-TEST", thread_id=thread_id,
    )

    result = graph.invoke(state, config=config)
    assert result["__interrupt__"], "expected the graph to pause at human_assessment_review"
    assert result["context"]["current_stage"] == "HUMAN_ASSESSMENT_REVIEW"

    resumed = graph.invoke(Command(resume={"decision": "APPROVE_COMPLETION_PLAN"}), config=config)
    assert resumed["context"]["status"] == "SUCCEEDED"
    assert resumed["context"]["current_stage"] == "COMPLETION_PLAN_READY"
    assert resumed["completion_plan_id"]


def test_run_resumes_from_a_fresh_graph_object_via_the_same_thread_id(tmp_path):
    """Proves persistence, not just in-process state: a brand-new compiled
    graph pointed at the same checkpoint db and thread_id must pick up exactly
    where the first one paused."""
    repo = make_repo(tmp_path)
    checkpoint_path = str(tmp_path / "checkpoints.sqlite")
    thread_id = "thread-test-0002"
    config = {"configurable": {"thread_id": thread_id}}

    first_graph = build_assessment_graph(repo, checkpoint_path)
    state = initial_state(
        source_path=str(tmp_path), project_id="PROJECT-TEST-2", run_id="RUN-TEST-2", thread_id=thread_id,
    )
    first_graph.invoke(state, config=config)

    second_graph = build_assessment_graph(repo, checkpoint_path)
    resumed = second_graph.invoke(Command(resume={"decision": "BLOCK_RUN"}), config=config)
    assert resumed["context"]["status"] == "BLOCKED"


def test_invalid_path_blocks_without_reaching_human_review(tmp_path):
    repo = make_repo(tmp_path)
    graph = build_assessment_graph(repo, str(tmp_path / "checkpoints.sqlite"))
    thread_id = "thread-test-0003"
    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(
        source_path="relative/not/absolute", project_id="PROJECT-TEST-3",
        run_id="RUN-TEST-3", thread_id=thread_id,
    )
    result = graph.invoke(state, config=config)
    assert result["context"]["status"] == "BLOCKED"
    assert not result.get("__interrupt__")


def test_manifest_records_real_files_with_checksums(tmp_path):
    (tmp_path / "results.csv").write_text("a,b\n1,2\n")
    (tmp_path / "notes.md").write_text("hello")

    repo = make_repo(tmp_path)
    graph = build_assessment_graph(repo, str(tmp_path / "checkpoints.sqlite"))
    thread_id = "thread-test-0004"
    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(
        source_path=str(tmp_path), project_id="PROJECT-TEST-4", run_id="RUN-TEST-4", thread_id=thread_id,
    )
    graph.invoke(state, config=config)

    rows = repo.conn.execute(
        "SELECT relative_path, checksum_sha256 FROM source_assets WHERE project_id = ?",
        ("PROJECT-TEST-4",),
    ).fetchall()
    scanned_paths = {row["relative_path"] for row in rows}
    assert "results.csv" in scanned_paths
    assert "notes.md" in scanned_paths
    assert all(row["checksum_sha256"] for row in rows)
