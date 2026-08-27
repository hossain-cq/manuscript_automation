# LangGraph research notes

Sources reviewed on 2026-08-21:

1. https://docs.langchain.com/oss/python/langgraph/overview
   - LangGraph is a low-level orchestration framework/runtime for long-running, stateful agents.
   - It supports mixing deterministic and LLM-driven steps.
   - The official overview highlights durable execution, streaming, human-in-the-loop, persistence, and debugging/observability integrations.
   - LangGraph can be used without LangChain; LangChain provides higher-level model/agent abstractions.

2. https://docs.langchain.com/oss/python/langgraph/persistence
   - Checkpointers persist thread-scoped graph state for continuity, human-in-the-loop, time travel, and fault tolerance.
   - Stores persist application-defined data across threads for long-term memory/shared knowledge.
   - Production should use a persistent checkpointer; in-memory persistence is not restart-safe.
   - Subgraph state propagation must be designed explicitly; shared state may require a store or parent-checkpoint configuration.

3. https://docs.langchain.com/oss/python/langgraph/interrupts
   - Dynamic interrupts pause execution, persist graph state, and wait for external input.
   - Resume uses the same thread_id and Command(resume=...).
   - Nodes may restart from the beginning when resumed, so side effects before interrupt must be idempotent or moved after the interrupt.
   - Interrupt payloads should be JSON-serializable.

4. https://docs.langchain.com/oss/python/langgraph/graph-api
   - Graphs are defined by State, Nodes, and Edges.
   - Nodes can be deterministic code or LLM-driven logic.
   - State channels use reducers to combine updates.
   - Typed input/output/private schemas are supported, but private channels can be exposed by broad streaming, so output filtering is required.

Architecture conclusion:
- Use LangGraph as the workflow/orchestration runtime and state-machine layer.
- Do not use graph state as the authoritative scientific database or artifact store.
- Keep raw/derived artifacts, claims, evidence, provenance, findings, and releases in an external structured persistence layer.
- Use interrupts for explicit human approvals and corrections.
- Make scientific side effects idempotent and preferably execute them in separate bounded job workers that return artifact references to LangGraph.
- Use subgraphs for discovery, audits, literature, manuscript production, and QA; define explicit input/output contracts rather than relying on implicit shared state.
- Add an application API, artifact store, policy engine, domain execution sandbox, and observability/evaluation layer around LangGraph.
