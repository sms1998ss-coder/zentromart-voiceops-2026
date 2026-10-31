"""Automated Practice Benchmark Runner for ZentroMart (questions.csv & conversations.csv)."""
import csv
import time
import os
from collections import defaultdict
from dotenv import load_dotenv
from src.rag import build_index
from src.agent import ZentroMartAgent

load_dotenv()

def check_must_include(reply: str, must_include: str) -> bool:
    if not must_include or not must_include.strip():
        return True
    reply_lower = reply.lower()
    for fact_group in must_include.split(" ; "):
        alternatives = [alt.strip().lower() for alt in fact_group.split(" / ") if alt.strip()]
        if alternatives and not any(alt in reply_lower for alt in alternatives):
            return False
    return True

def run_practice_benchmark(max_questions=10, max_conversations=3):
    os.makedirs("sample_outputs", exist_ok=True)
    retriever = build_index("./kb")
    agent = ZentroMartAgent(retriever=retriever)

    # 1. Evaluate Single-Turn Questions (Steps 1, 2 & 4)
    print(f"\n--- Running {max_questions} Single-Turn Practice Questions ---")
    with open("practice/questions.csv", encoding="utf-8", newline="") as f:
        questions = list(csv.DictReader(f))[:max_questions]

    q_results = []
    for row in questions:
        agent.reset_history()
        prompt = row["question"]
        reply, sources = agent.chat(prompt)
        passed_facts = check_must_include(reply, row["must_include"])
        print(f"[{row['id']}] Fact Check: {'PASS' if passed_facts else 'REVIEW'} | Gold: {row['gold_evidence']}")
        q_results.append({
            "id": row["id"],
            "question": prompt,
            "agent_reply": reply,
            "gold_evidence": row["gold_evidence"],
            "retrieved_sources": " | ".join(sources),
            "must_include": row["must_include"],
            "facts_matched": passed_facts,
        })
        time.sleep(4.5)

    with open("sample_outputs/practice_questions_results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(q_results[0].keys()))
        writer.writeheader()
        writer.writerows(q_results)

    # 2. Evaluate Multi-Turn Conversations (Step 3)
    print(f"\n--- Running {max_conversations} Multi-Turn Practice Conversations ---")
    grouped = defaultdict(list)
    with open("practice/conversations.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            grouped[r["conversation_id"]].append(r)

    c_results = []
    for cid in list(grouped.keys())[:max_conversations]:
        agent.reset_history()
        turns = sorted(grouped[cid], key=lambda x: int(x["turn"]))
        print(f"\nConversation {cid} ({turns[0]['scenario']}):")
        for t in turns:
            reply, sources = agent.chat(t["customer_message"])
            passed_facts = check_must_include(reply, t["must_include"])
            print(f"  Turn {t['turn']} | Expected Tool: {t['expected_tool_calls']} | Facts: {'PASS' if passed_facts else 'REVIEW'}")
            c_results.append({
                "conversation_id": cid,
                "turn": t["turn"],
                "customer_message": t["customer_message"],
                "agent_reply": reply,
                "expected_tool_calls": t["expected_tool_calls"],
                "must_include": t["must_include"],
                "facts_matched": passed_facts,
            })
            time.sleep(4.5)

    with open("sample_outputs/practice_conversations_results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(c_results[0].keys()))
        writer.writeheader()
        writer.writerows(c_results)

    print("\nSaved benchmark logs to sample_outputs/practice_*_results.csv")

if __name__ == "__main__":
    run_practice_benchmark(max_questions=10, max_conversations=3)