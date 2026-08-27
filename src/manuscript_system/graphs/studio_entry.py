from __future__ import annotations

"""Entrypoint for `langgraph dev` (LangGraph Studio's local dev server).

LangGraph CLI expects a module-level compiled graph object at the path given
in langgraph.json. build_assessment_graph() takes a Repository and a
checkpoint path rather than being a bare graph, so this module builds those
from the same settings the CLI uses and exposes the result as `graph`.
"""

from manuscript_system.graphs.assessment import build_assessment_graph
from manuscript_system.persistence.database import connect
from manuscript_system.persistence.repositories import Repository
from manuscript_system.settings import get_settings

settings = get_settings()
repo = Repository(connect(settings.database_path))
graph = build_assessment_graph(repo)
