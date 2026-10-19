"""Streamlit UI for Research Dead-End Detector."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from pipeline import FIELD_NAMES, run_pipeline

st.set_page_config(page_title="Research Dead-End Detector", page_icon="", layout="wide")
st.title("Research Dead-End Detector")
st.write(
    "Explore similar prior work and evidence-backed research directions for a project idea."
)

with st.sidebar:
    st.header("Live integration")
    st.caption("TinyFish search and fetch activity for the latest run.")
    call_log_placeholder = st.empty()

def render_call_log(call_log: dict[str, object]) -> None:
    with call_log_placeholder.container():
        st.write("Queries submitted", call_log.get("queries", 0))
        st.write("URLs fetched", call_log.get("urls_fetched", 0))
        st.write("URLs failed to fetch", len(call_log.get("fetch_errors", [])))
        st.write("Records analyzed", call_log.get("records", 0))
        for query in call_log.get("query_text", []):
            st.caption(f"Search: {query}")
        for url in call_log.get("fetched_urls", []):
            st.caption(f"Fetched: {url}")

render_call_log(st.session_state.get("call_log", {}))

idea = st.text_area(
    "Project idea",
    placeholder="e.g. offline crop disease detection from leaf photos",
    height=120,
)

if st.button("Find similar work", type="primary", disabled=not idea.strip()):
    progress = st.progress(0, text="Starting research pipeline...")
    step_order = ["Searching", "Fetching", "Extracting", "Ranking", "Analyzing"]

    def update_progress(step: str) -> None:
        progress.progress(
            (step_order.index(step) + 1) / len(step_order),
            text=f"{step}...",
        )

    try:
        result = run_pipeline(idea.strip(), progress_callback=update_progress)
    except Exception as error:
        st.warning(f"Live research could not complete ({error}). Showing cached demo data.")
        try:
            result = run_pipeline(idea.strip(), progress_callback=update_progress, use_demo=True)
        except Exception as demo_error:
            st.error(f"Demo data could not be loaded: {demo_error}")
            st.stop()

    progress.progress(1.0, text="Research complete")
    if result["demo"]:
        st.info(
            "Showing cached demo data. Sample summaries are illustrative; verify "
            "details against each linked source."
        )

    records = result["records"]
    call_log = {
        "queries": len(result["queries"]),
        "query_text": [query["query"] for query in result["queries"]],
        "urls_fetched": result["urls_fetched"],
        "fetched_urls": result["fetched_urls"],
        "fetch_errors": result.get("fetch_errors", []),
        "records": len(records),
    }
    st.session_state["call_log"] = call_log
    with st.sidebar:
        render_call_log(call_log)
    if call_log["fetch_errors"]:
        st.warning(
            f"TinyFish could not fetch {len(call_log['fetch_errors'])} source(s). "
            "Continuing with the pages that were available."
        )
        for fetch_error in call_log["fetch_errors"]:
            st.caption(f"{fetch_error['error']}: {fetch_error['url']}")

    st.subheader("Similar prior work")
    for record in records:
        with st.container(border=True):
            st.markdown(
                f"**[{record.get('title', 'Untitled source')}]({record.get('url', '')})**"
                f" · `{record.get('source_type', 'web')}`"
                f" · similarity **{record.get('score', 0.0):.3f}**"
            )
            for field in FIELD_NAMES:
                value = record.get(field, "unknown")
                evidence = record.get("evidence", {}).get(field, "")
                st.markdown(f"**{field.replace('_', ' ').title()}:** {value}")
                if evidence:
                    st.caption(f'Evidence: “{evidence}”')

    st.subheader("Method comparison")
    table = pd.DataFrame(
        [
            {
                "title": record.get("title", ""),
                **{field: record.get(field, "unknown") for field in FIELD_NAMES},
            }
            for record in records
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Crowded areas")
    method_counts = result["analysis"]["method_counts"]
    if method_counts:
        st.dataframe(
            pd.DataFrame(
                [{"method": method, "records": count} for method, count in method_counts.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No repeated methods were identified in the sources searched.")
    blind_spots = result["analysis"]["blind_spots"]
    if blind_spots:
        st.caption(
            "Often unspecified in these sources: "
            + ", ".join(field.replace("_", " ") for field in blind_spots)
        )

    st.subheader("Possible gaps")
    for item in result["analysis"]["hypotheses"]:
        st.markdown(f"- {item['hypothesis']}")
        st.caption("Sources: " + " · ".join(item["source_urls"]))

st.caption("Based on limited sources searched; not a novelty guarantee.")
