
"""
AI-Powered Visual Data Analytics and Business Intelligence Platform
Streamlit Dashboard - Milestone 4

Pages:
    Home
    PPE Detection
    Documents
    Ask AI
    Dashboard
    Report
"""

import os
import tempfile

import cv2
import streamlit as st

import knowledge_base as kb
import storage
import vision

from agent import ask_agent
from documents import process_file
from llm import MODEL, has_key, ask_llm
from report import build_pdf, write_summary


st.set_page_config(
    page_title="AI-Powered Visual Data Analytics and Business Intelligence Platform",
    page_icon="🦺",
    layout="wide",
)


# ----------------------------------------------------------
# GEMINI API KEY
# ----------------------------------------------------------

try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass


# ----------------------------------------------------------
# LOAD MODEL ONCE
# ----------------------------------------------------------

@st.cache_resource
def get_model():
    return vision.load_model()


@st.cache_resource
def load_sample_manual():
    """Load a sample document only when the knowledge base is empty."""
    if kb.total_chunks() == 0:
        for name in ["safety_manual.pdf", "sample_safety_manual.txt"]:
            path = os.path.join(vision.BASE_DIR, "data", name)

            if os.path.exists(path):
                chunks = process_file(path)

                if chunks:
                    kb.add_document(chunks, name)
                    return name

    return None


def save_upload(uploaded):
    """Save an uploaded file temporarily while preserving its extension."""
    suffix = os.path.splitext(uploaded.name)[1].lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as tmp:
        tmp.write(uploaded.getbuffer())
        return tmp.name


def process_uploaded_document(uploaded):
    """Process and index one uploaded document."""
    temp_path = save_upload(uploaded)

    try:
        chunks = process_file(temp_path)

        if not chunks:
            return 0

        return kb.add_document(chunks, uploaded.name)

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def make_document_answer(question, results):
    """Generate an answer grounded in retrieved document passages."""
    if not results:
        return (
            "Not found in this document. "
            "Try asking the question using different words."
        )

    context = "\n\n".join(
        f"[{item['source']} #{item['chunk']}] {item['text']}"
        for item in results
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the relevant passages provided below.

Instructions:
- Answer the exact question directly.
- Use simple, clear English.
- Do not repeat the entire document.
- Do not include unrelated sections or incident reports.
- If asked for rules, list only the rules relevant to the question.
- Keep the answer concise but include all relevant points present in the passages.
- Do not invent facts or safety requirements.
- If the passages do not contain the answer, say:
  "Not found in this document."
- Add source citations such as [safety_manual.pdf #2] to relevant points.

Document passages:
{context}

Question:
{question}

Answer:
"""

    reply = ask_llm(prompt)

    if not reply:
        return (
            "Gemini did not return an answer. "
            "Please check your API key and model configuration."
        )

    if reply.startswith("⚠️ Gemini error:"):
        return reply

    return reply


# ----------------------------------------------------------
# INITIALIZATION
# ----------------------------------------------------------

model = get_model()
load_sample_manual()


# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

with st.sidebar:
    st.markdown(
        "### 🦺 AI-Powered Visual Data Analytics "
        "and Business Intelligence Platform"
    )

    page = st.radio(
        "Go to",
        [
            "🏠 Home",
            "👷 PPE Detection",
            "📄 Documents",
            "🤖 Ask AI",
            "📊 Dashboard",
            "📋 Report",
        ],
    )

    st.divider()

    st.write(
        ("✅" if model else "❌")
        + " PPE model (best.pt)"
    )

    st.write(
        ("✅" if has_key() else "⚠️")
        + f" Gemini ({MODEL})"
    )

    st.write(
        f"📚 {kb.total_chunks()} chunks in knowledge base"
    )


# ==========================================================
# HOME
# ==========================================================

if page == "🏠 Home":
    st.title(
        "🦺 AI-Powered Visual Data Analytics "
        "and Business Intelligence Platform"
    )

    st.subheader(
        "VisionDesk AI - Multimodal Workplace Intelligence System"
    )

    st.markdown(
        """
1. **👷 PPE Detection** — upload a site image or video.
2. **📄 Documents** — upload and query safety manuals and reports.
3. **🤖 Ask AI** — ask questions using documents and camera findings.
4. **📊 Dashboard** — view compliance metrics and trends.
5. **📋 Report** — generate a downloadable safety report.
"""
    )

    if not model:
        st.warning(
            "Place your trained best.pt model in the expected project "
            "location to enable detection."
        )

    if not has_key():
        st.info(
            "Configure GEMINI_API_KEY in your .env file or Streamlit secrets "
            "to enable AI-generated answers."
        )


# ==========================================================
# PPE DETECTION - MILESTONE 1
# ==========================================================

elif page == "👷 PPE Detection":
    st.header("👷 PPE Detection")

    if not model:
        st.error("PPE model could not be loaded.")
        st.stop()

    kind = st.radio(
        "Input type",
        ["Image", "Video"],
        horizontal=True,
    )

    conf = st.slider(
        "Confidence",
        0.1,
        0.9,
        0.4,
        0.05,
        help="Higher confidence can reduce false detections but may miss objects.",
    )

    if kind == "Image":
        file = st.file_uploader(
            "Upload a site image",
            type=["jpg", "jpeg", "png"],
            key="ppe_image_upload",
        )

        if file and st.button("Analyze image", type="primary"):
            path = save_upload(file)

            try:
                image = cv2.imread(path)

                if image is None:
                    st.error("Could not read the uploaded image.")
                else:
                    dets = vision.detect(model, path, conf)
                    result = vision.summarize(dets, file.name)

                    storage.log_scan(file.name, result, "image")

                    st.session_state["last"] = {
                        "name": file.name,
                        "result": result,
                        "dets": dets,
                        "image": vision.draw(image, dets),
                    }
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass

    else:
        file = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov"],
            key="ppe_video_upload",
        )

        every_n = st.slider(
            "Check 1 of every N frames",
            1,
            30,
            5,
        )

        if file and st.button("Analyze video", type="primary"):
            path = save_upload(file)

            try:
                with st.spinner("Analyzing video..."):
                    dets, preview, frames = vision.detect_video(
                        model,
                        path,
                        every_n,
                        conf,
                    )

                result = vision.summarize(dets, file.name)
                storage.log_scan(file.name, result, "video")

                st.session_state["last"] = {
                    "name": f"{file.name} ({frames} frames)",
                    "result": result,
                    "dets": dets,
                    "image": preview,
                }
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass

    last = st.session_state.get("last")

    if last:
        col1, col2 = st.columns([3, 2])

        if last.get("image") is not None:
            col1.image(
                cv2.cvtColor(last["image"], cv2.COLOR_BGR2RGB),
                caption=last["name"],
            )

        with col2:
            result = last["result"]

            if result.get("compliant"):
                st.success("✅ COMPLIANT — no violations detected")
            else:
                st.error(
                    "🚨 VIOLATIONS: "
                    + ", ".join(
                        sorted(set(result.get("violations", [])))
                    )
                )

            st.metric("Objects detected", len(last.get("dets", [])))
            st.metric(
                "Violations",
                len(result.get("violations", [])),
            )
            st.write("**Counts:**", result.get("counts", {}))
            st.caption("This result can be used in Ask AI.")


# ==========================================================
# DOCUMENTS - MILESTONE 2
# ==========================================================

elif page == "📄 Documents":
    st.header("📄 Document Knowledge Base")

    st.write(
        "Upload a document, then enter a question below. "
        "The question will be answered using the active document."
    )

    # One upload area for PDFs, text files and Word documents.
    files = st.file_uploader(
        "Upload manuals / inspection or incident reports",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="document_upload",
    )

    if files and st.button(
        "Add to knowledge base",
        type="primary",
        key="add_documents_button",
    ):
        successfully_added = []

        for uploaded in files:
            try:
                with st.spinner(f"Processing {uploaded.name}..."):
                    count = process_uploaded_document(uploaded)

                if count:
                    successfully_added.append(uploaded.name)
                    st.success(
                        f"{uploaded.name}: {count} chunks indexed."
                    )
                else:
                    st.warning(
                        f"{uploaded.name}: no readable text was found. "
                        "Scanned PDFs may need OCR."
                    )

            except Exception as exc:
                st.error(
                    f"Could not process {uploaded.name}: {exc}"
                )

        # Automatically make the last successfully processed file active.
        if successfully_added:
            st.session_state["active_document"] = successfully_added[-1]

    st.subheader("Documents in the knowledge base")

    docs = kb.list_documents()

    if docs:
        st.table(
            [
                {"Document": name, "Chunks": count}
                for name, count in docs.items()
            ]
        )
    else:
        st.info("Upload and add a document to get started.")

    st.divider()

    # No second document selector: use the active document automatically.
    st.subheader("🔍 Ask a question about a document")

    if docs:
        document_names = list(docs.keys())

        active_document = st.session_state.get("active_document")

        if active_document not in document_names:
            active_document = document_names[-1]
            st.session_state["active_document"] = active_document

        st.caption(f"Active document: **{active_document}**")

        question = st.text_input(
            "Enter your question",
            placeholder="What are the workplace safety rules?",
            key="document_question",
        )

        if st.button(
            "Find answer",
            type="primary",
            key="find_document_answer",
        ):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Finding relevant passages and preparing an answer..."):
                    try:
                        results = kb.search(
                            question,
                            top_k=5,
                            source=active_document,
                        )

                        answer = make_document_answer(question, results)

                        st.session_state["document_qa_result"] = {
                            "question": question,
                            "document": active_document,
                            "answer": answer,
                            "results": results,
                        }

                    except Exception as exc:
                        st.error(f"Document search failed: {exc}")

        qa_result = st.session_state.get("document_qa_result")

        if (
            qa_result
            and qa_result.get("document") == active_document
        ):
            st.subheader("Answer")
            st.markdown(qa_result["answer"])

            with st.expander("View retrieved source passages"):
                for item in qa_result["results"]:
                    st.caption(
                        f"{item['source']} #{item['chunk']} "
                        f"(distance: {item['distance']})"
                    )
                    st.write(item["text"])
                    st.divider()

    else:
        st.info("Upload and add a document before asking a question.")


# ==========================================================
# ASK AI - MILESTONE 3
# ==========================================================

elif page == "🤖 Ask AI":
    st.header("🤖 Ask the Safety AI")

    last = st.session_state.get("last")
    use_image = False

    if last:
        use_image = st.checkbox(
            f"Include last detection result ({last['name']})",
            value=True,
        )

        if use_image:
            st.caption(last["result"].get("text", ""))

    question = st.text_input(
        "Your question",
        placeholder="Is this worker compliant? Which rule applies?",
        key="agent_question",
    )

    if st.button("Ask", type="primary", key="ask_agent_button"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            vision_result = (
                last["result"]
                if use_image and last
                else None
            )

            with st.spinner("Thinking..."):
                try:
                    output = ask_agent(
                        question,
                        vision=(
                            vision_result.get("text", "")
                            if vision_result
                            else ""
                        ),
                        violations=(
                            vision_result.get("violations", [])
                            if vision_result
                            else []
                        ),
                    )

                    st.session_state["answer"] = {
                        "question": question,
                        **output,
                    }
                    st.session_state["voted"] = False

                except Exception as exc:
                    st.error(f"Could not generate an answer: {exc}")

    ans = st.session_state.get("answer")

    if ans:
        st.subheader("Answer")
        st.markdown(ans.get("answer", ""))

        with st.expander("How the agent worked"):
            st.write(
                "**Steps:** "
                + " → ".join(ans.get("steps", []))
            )

            for item in ans.get("docs", []):
                st.caption(
                    f"[{item['source']} #{item['chunk']}] "
                    f"{item['text'][:300]}"
                )

        if not st.session_state.get("voted", False):
            st.write("Was this answer helpful?")

            col1, col2 = st.columns(2)

            if col1.button("👍 Yes", key="feedback_yes"):
                storage.log_feedback(ans["question"], True)
                st.session_state["voted"] = True
                st.rerun()

            if col2.button("👎 No", key="feedback_no"):
                storage.log_feedback(ans["question"], False)
                st.session_state["voted"] = True
                st.rerun()

        else:
            st.caption("Thanks for your feedback!")


# ==========================================================
# DASHBOARD - MILESTONE 4
# ==========================================================

elif page == "📊 Dashboard":
    st.header("📊 Safety Dashboard")

    scans = storage.load_scans()
    kpis = storage.kpis(scans)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Images / videos checked",
        kpis["scans"],
    )

    c2.metric(
        "Compliance rate",
        (
            f"{kpis['compliance']}%"
            if kpis["compliance"] is not None
            else "-"
        ),
    )

    c3.metric(
        "Total violations",
        kpis["violations"],
    )

    c4.metric(
        "User satisfaction",
        (
            f"{kpis['satisfaction']}%"
            if kpis["satisfaction"] is not None
            else "-"
        ),
        help="Target: 85% or more",
    )

    if scans.empty:
        st.info("No scan data yet. Analyze images or videos in PPE Detection.")

    else:
        left, right = st.columns(2)

        left.subheader("Violations by type")
        counts = storage.violation_counts(scans)

        if len(counts):
            left.bar_chart(counts)
        else:
            left.success("No violations recorded.")

        right.subheader("Violations per day")
        right.line_chart(
            scans.set_index("time")["violations"].resample("D").sum()
        )

        st.subheader("All scans")
        st.dataframe(
            scans.sort_values("time", ascending=False),
            hide_index=True,
        )

    if st.button("🗑️ Clear history"):
        storage.clear_all()
        st.rerun()


# ==========================================================
# REPORT - MILESTONE 4
# ==========================================================

elif page == "📋 Report":
    st.header("📋 Compliance Report")

    scans = storage.load_scans()

    if st.button("Generate report", type="primary"):
        kpis = storage.kpis(scans)
        counts = storage.violation_counts(scans)

        with st.spinner("Preparing report..."):
            summary = write_summary(kpis, counts)

            st.session_state["report"] = {
                "summary": summary,
                "pdf": build_pdf(
                    kpis,
                    counts,
                    scans,
                    summary,
                ),
            }

    report_data = st.session_state.get("report")

    if report_data:
        st.subheader("Summary")
        st.write(report_data["summary"])

        st.download_button(
            "⬇️ Download PDF",
            report_data["pdf"],
            file_name="safety_compliance_report.pdf",
            mime="application/pdf",
        )

        st.download_button(
            "⬇️ Download data (CSV)",
            scans.to_csv(index=False),
            file_name="scans.csv",
            mime="text/csv",
        )
