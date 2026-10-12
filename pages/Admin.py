"""Password-protected FAQ and feedback administration."""

import hmac
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from chat_helpers import cluster_unanswered_questions, environment_flag
import database

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

st.set_page_config(page_title="FAQ Admin", page_icon="🛠️")
st.title("FAQ administration")

admin_password = os.getenv("ADMIN_PASSWORD")
if not admin_password:
    st.error("Set ADMIN_PASSWORD in your .env file to enable the admin page.")
    st.stop()

if not st.session_state.get("admin_authenticated", False):
    entered_password = st.text_input("Admin password", type="password")
    if st.button("Sign in"):
        if hmac.compare_digest(entered_password, admin_password):
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect admin password.")
    st.stop()

try:
    faqs = database.load_faqs()
except Exception as error:
    st.error("The FAQ file could not be loaded.")
    st.exception(error)
    st.stop()


def save_and_reindex(
    updated_faqs: list[dict[str, str]],
    unanswered_ids: list[int] | None = None,
) -> None:
    """Validate, persist and incrementally reindex changed FAQs."""
    try:
        database.save_faqs(updated_faqs)
    except Exception as error:
        st.error("The FAQ changes could not be saved.")
        st.exception(error)
        return

    try:
        summary = database.create_embeddings()
    except Exception as error:
        st.error(
            "The FAQ changes were saved, but reindexing failed. "
            "Use Reindex FAQs to retry."
        )
        st.exception(error)
        return

    if unanswered_ids:
        try:
            database.delete_unanswered_questions(unanswered_ids)
        except Exception as error:
            st.error(
                "The FAQ was indexed, but the unanswered questions could not "
                "be removed."
            )
            st.exception(error)
            return

    st.success(
        "FAQ index updated: "
        f"{summary['added']} added, {summary['updated']} updated, "
        f"{summary['removed']} removed, {summary['unchanged']} unchanged."
    )
    st.session_state.pop("_unanswered_faq_question", None)
    st.session_state.pop("_unanswered_faq_ids", None)
    st.rerun()


st.subheader("Manage FAQs")
prefill_question = st.session_state.get("_unanswered_faq_question", "")
prefill_unanswered_ids = st.session_state.get("_unanswered_faq_ids", [])
with st.expander("Add FAQ", expanded=bool(prefill_question)):
    with st.form("add_faq_form"):
        new_question = st.text_input("Question", value=prefill_question)
        new_answer = st.text_area("Answer")
        add_submitted = st.form_submit_button("Add and reindex")
    if add_submitted:
        try:
            updated_faqs = database.validate_faqs(
                [*faqs, {"question": new_question, "answer": new_answer}]
            )
        except ValueError as error:
            st.error(str(error))
        else:
            save_and_reindex(updated_faqs, prefill_unanswered_ids)

if faqs:
    faq_labels = [
        f"{index + 1}. {faq['question']}"
        for index, faq in enumerate(faqs)
    ]
    selected_label = st.selectbox("Select FAQ to edit", faq_labels)
    selected_index = faq_labels.index(selected_label)

    with st.form("edit_faq_form"):
        edited_question = st.text_input(
            "Question",
            value=faqs[selected_index]["question"],
        )
        edited_answer = st.text_area(
            "Answer",
            value=faqs[selected_index]["answer"],
        )
        edit_submitted = st.form_submit_button("Save changes and reindex")
    if edit_submitted:
        updated_faqs = [faq.copy() for faq in faqs]
        updated_faqs[selected_index] = {
            "question": edited_question,
            "answer": edited_answer,
        }
        try:
            updated_faqs = database.validate_faqs(updated_faqs)
        except ValueError as error:
            st.error(str(error))
        else:
            save_and_reindex(updated_faqs)

    with st.form("delete_faq_form"):
        st.caption(f"Selected FAQ: {faq_labels[selected_index]}")
        delete_submitted = st.form_submit_button("Delete selected FAQ")
    if delete_submitted:
        updated_faqs = [
            faq.copy() for index, faq in enumerate(faqs)
            if index != selected_index
        ]
        try:
            updated_faqs = database.validate_faqs(updated_faqs)
        except ValueError as error:
            st.error(str(error))
        else:
            save_and_reindex(updated_faqs)

if st.button("Reindex FAQs"):
    try:
        summary = database.create_embeddings()
    except Exception as error:
        st.error("The FAQ index could not be updated.")
        st.exception(error)
    else:
        st.success(
            "FAQ index updated: "
            f"{summary['added']} added, {summary['updated']} updated, "
            f"{summary['removed']} removed, {summary['unchanged']} unchanged."
        )

st.subheader("Feedback")
if not environment_flag("SHOW_FEEDBACK"):
    st.caption("Feedback is disabled.")
else:
    try:
        feedback_stats = database.get_feedback_stats()
        stat_columns = st.columns(3)
        stat_columns[0].metric("Total", feedback_stats["total"])
        stat_columns[1].metric("Positive", feedback_stats["positive"])
        stat_columns[2].metric("Negative", feedback_stats["negative"])
    except Exception as error:
        st.error("Feedback statistics could not be loaded.")
        st.exception(error)

unanswered_tab, analytics_tab = st.tabs(["Unanswered questions", "Analytics"])

with unanswered_tab:
    st.subheader("Unanswered question clusters")
    try:
        unanswered = database.get_unanswered_questions()
        if not unanswered:
            st.info("No unanswered questions yet.")
        else:
            unanswered.sort(key=lambda row: int(row["id"]))
            questions = [str(row["question"]) for row in unanswered]
            embeddings = list(
                database.get_embedding_model().query_embed(questions)
            )
            threshold = float(
                os.getenv("UNANSWERED_CLUSTER_THRESHOLD", "0.8")
            )
            clusters = cluster_unanswered_questions(
                unanswered,
                embeddings,
                threshold,
            )
            for index, cluster in enumerate(clusters):
                st.markdown(
                    f"**Cluster {index + 1} — {cluster['count']} questions**"
                )
                for example in cluster["example_questions"]:
                    st.write(f"- {example}")
                first_row = cluster["rows"][0]
                if st.button(
                    "Add as FAQ",
                    key=f"add_unanswered_cluster_{index}",
                ):
                    st.session_state["_unanswered_faq_question"] = str(
                        first_row["question"]
                    )
                    st.session_state["_unanswered_faq_ids"] = [
                        int(row["id"]) for row in cluster["rows"]
                    ]
                    st.rerun()
    except Exception as error:
        st.error("Unanswered questions could not be clustered.")
        st.exception(error)

with analytics_tab:
    st.subheader("Analytics")
    try:
        analytics = database.get_analytics_stats()
        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "Fallback rate",
            f"{analytics['fallback_rate']:.1%}",
        )
        metric_columns[1].metric(
            "Conversations",
            analytics["total_conversations"],
        )
        metric_columns[2].metric("Messages", analytics["total_messages"])
        metric_columns[3].metric(
            "Unanswered",
            analytics["total_unanswered"],
        )

        daily_questions = analytics["questions_per_day"]
        if daily_questions:
            st.line_chart(
                daily_questions,
                x="date",
                y="questions",
            )
        else:
            st.info("No questions have been asked yet.")

        st.markdown("**Top 10 asked questions**")
        top_questions = analytics["top_questions"]
        if top_questions:
            st.dataframe(
                top_questions,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No question data is available yet.")
    except Exception as error:
        st.error("Analytics could not be loaded from the database.")
        st.exception(error)
