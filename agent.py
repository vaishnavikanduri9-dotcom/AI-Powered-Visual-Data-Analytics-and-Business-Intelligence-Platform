PROMPT = """You are a careful document question-answering assistant.

Answer the user's exact question using only the retrieved document passages.

Instructions:
1. Give a direct answer first.
2. If the question asks for rules, steps, or a list, include all relevant items found in the passages.
3. Do not mix unrelated sections.
4. Do not invent facts, rules, deadlines, or requirements.
5. If the retrieved passages do not contain the answer, say:
   "Not found in the retrieved document passages."
6. Cite document-based facts using [filename #chunk].
7. If the source is unclear or incomplete, say so.
8. Treat camera findings as observations, not proof of a violation.

Camera findings:
{vision}

Retrieved document passages:
{docs}

Question: {question}

Answer:"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from knowledge_base import search
from llm import ask_llm

PROMPT = """You are a workplace safety assistant.
Answer using ONLY the documents below. If the answer is not there, say "Not found in documents."
Mention the source of each fact like [file #chunk].
If camera findings are given, say whether they break any rule.

Camera findings: {vision}

Documents:
{docs}

Question: {question}
Answer:"""


class State(TypedDict, total=False):
    question: str
    vision: str          # e.g. "site.jpg: VIOLATIONS: ['no_helmet']"
    violations: list
    search_query: str
    docs: list
    answer: str
    steps: list


# ── Step 1 (optional): use the camera result to improve the search ──
def use_image(state):
    query = state["question"] + " " + " ".join(set(state.get("violations", [])))
    return {"search_query": query, "steps": state.get("steps", []) + ["use_image"]}


# ── Step 2: find relevant document chunks ──

def retrieve(state):
    query = state.get("search_query") or state["question"]

    docs = search(query, top_k=5)

    return {
        "docs": docs,
        "steps": state.get("steps", []) + ["retrieve"],
    }



# ── Step 3: ask Gemini, grounded in those chunks ──
def answer(state):
    docs_text = "\n\n".join(f"[{d['source']} #{d['chunk']}] {d['text']}" for d in state["docs"])
    prompt = PROMPT.format(vision=state.get("vision") or "None",
                           docs=docs_text or "(no documents)",
                           question=state["question"])
    reply = ask_llm(prompt)

    if reply is None:   # no API key -> show the best passages instead of crashing
        reply = "⚠️ GEMINI_API_KEY not set. Most relevant passages:\n\n" + "\n\n".join(
            f"- [{d['source']} #{d['chunk']}] {d['text']}" for d in state["docs"])

    return {"answer": reply, "steps": state.get("steps", []) + ["answer"]}


def route(state):
    return "use_image" if state.get("vision") else "retrieve"


graph = StateGraph(State)
graph.add_node("use_image", use_image)
graph.add_node("retrieve", retrieve)
graph.add_node("answer", answer)
graph.add_conditional_edges(START, route, {"use_image": "use_image", "retrieve": "retrieve"})
graph.add_edge("use_image", "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)
agent = graph.compile()


def ask_agent(question, vision="", violations=None):
    return agent.invoke({"question": question, "vision": vision,
                         "violations": violations or [], "steps": []})


if __name__ == "__main__":
    result = ask_agent("What PPE is required on site?")
    print("Steps:", result["steps"])
    print(result["answer"])
