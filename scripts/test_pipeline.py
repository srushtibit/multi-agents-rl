#!/usr/bin/env python3
"""
End-to-end smoke test:
- Index the dataset directory into the unified knowledge base
- Spin up agents and coordinator
- Send a sample user query
- Run a few coordination cycles and print the response
"""

import asyncio
from pathlib import Path
import sys


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


async def main():
    # Ensure project root on path
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Lazy imports after sys.path
    from kb.unified_knowledge_base import get_knowledge_base
    from agents.base_agent import AgentCoordinator, Message, MessageType
    from agents.communication.communication_agent import CommunicationAgent
    from agents.retrieval.retrieval_agent import RetrievalAgent
    from agents.critic.critic_agent import CriticAgent
    from agents.escalation.escalation_agent import EscalationAgent

    print("\n=== Building Knowledge Base ===")
    kb = get_knowledge_base()
    dataset_dir = root / "dataset"
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return 1

    success, total = kb.add_documents_from_directory(str(dataset_dir))
    print(f"Indexed {success}/{total} documents from {dataset_dir}")

    stats = kb.get_stats()
    print(f"KB Chunks: {stats.total_chunks}, Docs: {stats.total_documents}, Languages: {stats.languages}")

    print("\n=== Spinning up Agents ===")
    coordinator = AgentCoordinator()
    comm = CommunicationAgent()
    retr = RetrievalAgent()
    crit = CriticAgent()
    esc = EscalationAgent()

    coordinator.register_agent(comm)
    coordinator.register_agent(retr)
    coordinator.register_agent(crit)
    coordinator.register_agent(esc)

    coordinator.start_all_agents()

    # Prepare user message
    query_text = "Where can I get training on the new software system?"
    user_msg = Message(
        type=MessageType.QUERY,
        content=query_text,
        sender="user",
        recipient="communication_agent",
        language="en",
    )

    # Inject to communication agent
    comm.receive_message(user_msg)

    print("\n=== Running coordination cycles ===")
    final_response = None
    evaluation = None

    for cycle in range(10):
        messages = await coordinator.run_cycle()
        if not messages:
            continue
        for m in messages:
            if m is None:
                continue
            if m.type == MessageType.RESPONSE and m.sender == "retrieval_agent":
                final_response = m.content
            if m.type == MessageType.FEEDBACK and hasattr(m, "metadata"):
                evaluation = m.metadata.get("evaluation_result")
        if final_response:
            break

    print("\n=== Result ===")
    if final_response:
        print(final_response)
    else:
        print("No response generated")
    if evaluation:
        print("\nEvaluation:")
        print(evaluation)

    coordinator.stop_all_agents()
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))


