"""Streamlit presentation layer for the project intelligence workspace."""

from datetime import datetime

import altair as alt
import streamlit as st

from document_ingestion import (
    document_content_hash,
    ingest_uploaded_documents,
)
from project_agents import (
    calculate_project_health,
    generate_project_documentation,
)
from rag_pipeline import (
    DEFAULT_EMBEDDING_MODEL,
    SQLiteVectorStore,
    answer_project_question,
)


KNOWLEDGE_BASE_PATH = ".data/project_knowledge_base.sqlite3"
NAVIGATION = (
    ":material/dashboard:  Dashboard",
    ":material/folder_open:  Document intelligence",
    ":material/precision_manufacturing:  Project analysis agents",
    ":material/shield:  Risk analysis",
    ":material/monitor_heart:  Project health",
    ":material/description:  Documentation generator",
    ":material/assistant:  AI project assistant",
    ":material/settings:  Settings",
)
PAGE_LABELS = {
    NAVIGATION[0]: "Dashboard",
    NAVIGATION[1]: "Document intelligence",
    NAVIGATION[2]: "Project analysis agents",
    NAVIGATION[3]: "Risk analysis",
    NAVIGATION[4]: "Project health",
    NAVIGATION[5]: "Documentation generator",
    NAVIGATION[6]: "AI project assistant",
    NAVIGATION[7]: "Settings",
}
SUGGESTED_QUESTIONS = (
    "What is the current project status?",
    "What are the biggest project risks?",
    "What tasks are pending?",
    "Are we on track?",
)


@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(DEFAULT_EMBEDDING_MODEL)


@st.cache_resource
def load_vector_store():
    return SQLiteVectorStore(KNOWLEDGE_BASE_PATH)


def _initialize_session_state():
    defaults = {
        "project_name": "AI Project Intelligence & Risk Advisor",
        "project_analyses": [],
        "project_chat_history": [],
        "pending_chat_question": None,
        "generated_documentation": None,
        "generated_report": None,
        "recent_activity": [],
        "last_uploaded_names": [],
        "dashboard_search_results": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _apply_styles():
    st.markdown(
        """
        <style>
        :root {
            --navy: #e8f0ff;
            --blue: #4b8cff;
            --purple: #8b73ff;
            --ink: #e4ecfb;
            --muted: #93a4c2;
            --line: #243552;
            --surface: #101d32;
            --wash: #0a1426;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 82% 0%, rgba(52, 105, 206, .14), transparent 35%),
                linear-gradient(180deg, #0a1426 0%, #0c172a 100%);
            color: var(--ink);
        }
        [data-testid="stHeader"] { background: rgba(10, 20, 38, .82); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #101c31 0%, #111f37 100%);
            border-right: 1px solid rgba(147, 164, 194, .15);
        }
        [data-testid="stSidebar"] * { color: #e7efff; }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #c8d4ed;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding: .62rem .7rem;
            border-radius: .55rem;
            transition: background .18s ease, transform .18s ease;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(101, 143, 226, .13);
            transform: translateX(2px);
        }
        .block-container {
            max-width: 1440px;
            padding-top: 1.65rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 { color: var(--navy); letter-spacing: -.025em; }
        h1 { font-weight: 750; }
        h2, h3 { font-weight: 680; }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, #142744, #101c31 78%);
            border: 1px solid var(--line);
            border-radius: 17px;
            padding: 1rem 1.15rem;
            box-shadow: 0 8px 26px rgba(0, 0, 0, .16);
            transition: transform .18s ease, box-shadow .18s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, .24);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: #f0f5ff; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(155deg, rgba(17, 31, 53, .96), rgba(14, 26, 45, .96));
            border-color: #243653;
            border-radius: 16px;
            box-shadow: 0 8px 26px rgba(0, 0, 0, .12);
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 10px;
            border: 1px solid #304464;
            font-weight: 620;
            transition: transform .16s ease, box-shadow .16s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 5px 16px rgba(57, 116, 216, .22);
            border-color: #5b86d2;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(105deg, #397fe8, #735ff0);
            border: 0;
            color: white;
        }
        [data-testid="stExpander"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 14px;
        }
        [data-testid="stChatMessage"] {
            border: 1px solid var(--line);
            border-radius: 16px;
            background: var(--surface);
            padding: .85rem 1rem;
        }
        [data-testid="stFileUploader"] section {
            border-radius: 14px;
            border-color: #425f8d;
            background: rgba(16, 29, 50, .96);
        }
        [data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, var(--blue), var(--purple));
        }
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            background: #0c1729;
            border-color: #2a3d5d;
            color: #edf3ff;
        }
        [data-testid="stCaptionContainer"] { color: var(--muted); }
        .hero-kicker {
            color: #8eb6ff;
            font-size: .79rem;
            font-weight: 700;
            letter-spacing: .13em;
            text-transform: uppercase;
        }
        .hero-copy {
            color: #aab9d2;
            max-width: 780px;
            font-size: 1.03rem;
        }
        @media (max-width: 760px) {
            .block-container { padding: 1.15rem .85rem 2rem; }
            h1 { font-size: 1.8rem; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                transition-duration: .01ms !important;
                animation-duration: .01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _analysis_totals(analyses):
    risks = [
        risk
        for analysis in analyses
        for risk in analysis["delivery_risk"]["risks"]
    ]
    blockers = [
        blocker
        for analysis in analyses
        for blocker in analysis["blockers_and_actions"]["blockers"]
    ]
    decisions = [
        decision
        for analysis in analyses
        for decision in analysis["blockers_and_actions"]["pending_decisions"]
    ]
    actions = [
        action
        for analysis in analyses
        for action in analysis["blockers_and_actions"]["actions"]
    ]
    findings = [
        finding
        for analysis in analyses
        for finding in analysis["scope"]["findings"]
    ]
    return {
        "risks": risks,
        "blockers": blockers,
        "decisions": decisions,
        "actions": actions,
        "findings": findings,
        "milestones": [
            finding
            for finding in findings
            if "milestone" in finding["categories"]
        ],
        "high_risks": sum(risk["severity"] == "high" for risk in risks),
        "medium_risks": sum(risk["severity"] == "medium" for risk in risks),
    }


def _record_activity(kind, detail, source=None):
    events = st.session_state.recent_activity
    events.insert(
        0,
        {
            "kind": kind,
            "detail": detail,
            "source": source,
            "timestamp": datetime.now().astimezone().strftime("%b %d, %Y · %I:%M %p"),
        },
    )
    del events[20:]


def _sync_upload_activity(uploaded_files):
    current_names = [uploaded_file.name for uploaded_file in uploaded_files or []]
    previous_names = st.session_state.last_uploaded_names
    for name in current_names:
        if name not in previous_names:
            _record_activity("Document added", "Added to this workspace", name)
    st.session_state.last_uploaded_names = current_names


def _ingest_uploads(uploaded_files, store, force=False):
    pending_files = []
    for uploaded_file in uploaded_files or []:
        try:
            is_unchanged = (
                not force
                and store.get_document_hash(uploaded_file.name)
                == document_content_hash(uploaded_file)
            )
        except Exception as error:
            st.error(f"Could not inspect {uploaded_file.name}: {error}")
            continue
        if not is_unchanged:
            pending_files.append(uploaded_file)

    if not pending_files:
        return

    try:
        model = load_embedding_model()
    except Exception as error:
        st.error(f"Could not initialize the embedding model: {error}")
        return

    with st.status(
        f"Updating the knowledge base with {len(pending_files)} document(s)…",
        expanded=False,
    ) as status:
        result = ingest_uploaded_documents(pending_files, store, model, force=force)
        processed = result["processed"]
        errors = result["errors"]

        for item in processed:
            _record_activity(
                "Document indexed",
                f"Indexed {item['chunk_count']} searchable chunks and refreshed analysis",
                item["source"],
            )

        if processed:
            st.session_state.project_analyses = store.list_document_analyses()
            st.session_state.generated_documentation = None
            st.session_state.generated_report = None
            status.update(
                label=f"Knowledge base updated with {len(processed)} document(s)",
                state="complete",
            )
        elif errors:
            status.update(label="Knowledge base update failed", state="error")
        else:
            status.update(label="Knowledge base is up to date", state="complete")

    if processed:
        chunk_count = sum(item["chunk_count"] for item in processed)
        st.success(
            f"Indexed {len(processed)} changed document(s) "
            f"({chunk_count} searchable chunk(s))."
        )
    for item in errors:
        st.error(f"Could not process {item['source']}: {item['error']}")


def _risk_counts(risks):
    return {
        severity: sum(risk["severity"] == severity for risk in risks)
        for severity in ("high", "medium")
    }


def _health_ring(score):
    ring = alt.Chart(
        alt.Data(
            values=[
                {"segment": "Health score", "value": score},
                {"segment": "Remaining", "value": 100 - score},
            ]
        )
    ).mark_arc(innerRadius=60, outerRadius=78).encode(
        theta=alt.Theta("value:Q"),
        color=alt.Color(
            "segment:N",
            scale=alt.Scale(
                domain=["Health score", "Remaining"],
                range=["#5b9aff", "#263854"],
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("segment:N", title="Measure"),
            alt.Tooltip("value:Q", title="Points"),
        ],
    )
    label = alt.Chart(
        alt.Data(values=[{"score": f"{score}"}])
    ).mark_text(
        fontSize=26,
        fontWeight=700,
        color="#f0f5ff",
    ).encode(
        x=alt.value(100),
        y=alt.value(93),
        text="score:N",
    )
    return (
        alt.layer(ring, label)
        .properties(width=200, height=186)
        .configure_view(stroke=None)
    )


def _risk_donut(risks):
    counts = _risk_counts(risks)
    values = [
        {"severity": severity.title(), "count": count}
        for severity, count in counts.items()
        if count
    ]
    if not values:
        return None
    return (
        alt.Chart(alt.Data(values=values))
        .mark_arc(innerRadius=58, outerRadius=82)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color(
                "severity:N",
                scale=alt.Scale(
                    domain=["High", "Medium"],
                    range=["#f28a52", "#e7bb55"],
                ),
                legend=alt.Legend(
                    title=None,
                    labelColor="#d7e3f7",
                    orient="bottom",
                ),
            ),
            tooltip=[
                alt.Tooltip("severity:N", title="Severity"),
                alt.Tooltip("count:Q", title="Signals"),
            ],
        )
        .properties(height=240)
        .configure_view(stroke=None)
    )


def _dashboard_recommendations(analyses, totals):
    if not analyses:
        return []

    recommendations = []
    for risk in totals["risks"]:
        related_actions = [
            action
            for action in totals["actions"]
            if action["source"] == risk["source"]
        ]
        response = (
            f"Documented follow-up: {related_actions[0]['text']}"
            if related_actions
            else "No mitigation action is documented in this source; assign an owner and record a response."
        )
        recommendations.append(
            {
                "priority": risk["severity"].title(),
                "text": f"Review this delivery risk: {risk['evidence']}",
                "detail": response,
                "source": risk["source"],
            }
        )

    for blocker in totals["blockers"]:
        recommendations.append(
            {
                "priority": "High",
                "text": f"Resolve the documented blocker: {blocker['text']}",
                "detail": "Confirm an owner and record the resolution in the project source documents.",
                "source": blocker["source"],
            }
        )

    for decision in totals["decisions"]:
        recommendations.append(
            {
                "priority": "Medium",
                "text": f"Close the pending decision: {decision['text']}",
                "detail": "Capture the decision and its owner in the project documentation.",
                "source": decision["source"],
            }
        )

    categories = {
        category
        for analysis in analyses
        for finding in analysis["scope"]["findings"]
        for category in finding["categories"]
    }
    for category, label in (
        ("goal", "project goal"),
        ("deliverable", "deliverable"),
        ("milestone", "milestone"),
        ("timeline", "timeline or due date"),
    ):
        if category not in categories:
            recommendations.append(
                {
                    "priority": "Medium",
                    "text": f"Document the missing {label}.",
                    "detail": "This scope signal was not found in the analyzed documents.",
                    "source": None,
                }
            )
    return recommendations[:6]


def _source_caption(source):
    st.caption(f"Source document · {source}")


def _render_items(items, text_key, empty_message, detail_builder=None):
    if not items:
        st.info(empty_message)
        return
    for item in items:
        with st.container(border=True):
            st.write(item[text_key])
            if detail_builder:
                details = detail_builder(item)
                if details:
                    st.caption(" · ".join(details))
            if item.get("source"):
                _source_caption(item["source"])


def _render_dashboard(analyses, store, project_name):
    totals = _analysis_totals(analyses)
    if analyses:
        health = calculate_project_health(analyses)
        score = health["overall_score"]
    else:
        health = None

    title_column, activity_column = st.columns([5, 1], vertical_alignment="center")
    title_column.markdown(
        '<div class="hero-kicker">Project intelligence workspace</div>',
        unsafe_allow_html=True,
    )
    title_column.title(f"Welcome back, {project_name}")
    activity_column.badge(
        f"{len(st.session_state.recent_activity)} activity",
        icon=":material/notifications:",
    )
    st.markdown(
        '<div class="hero-copy">A clear, evidence-led view of project scope, '
        'delivery risks, actions, and health — grounded in your uploaded documents.</div>',
        unsafe_allow_html=True,
    )
    with st.form("dashboard_search", border=False):
        search_column, button_column = st.columns([6, 1])
        query = search_column.text_input(
            "Search project information",
            placeholder="Search project documents, risks, milestones...",
            type="search",
            label_visibility="collapsed",
        )
        submitted = button_column.form_submit_button(
            "Search", icon=":material/search:", type="primary"
        )
    if submitted:
        if not query.strip():
            st.warning("Enter a phrase to search the indexed project documents.")
            st.session_state.dashboard_search_results = None
        elif store.chunk_count() == 0:
            st.info("Index project documents before searching.")
            st.session_state.dashboard_search_results = None
        else:
            try:
                model = load_embedding_model()
                embedding = model.encode(
                    [query],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )[0]
                st.session_state.dashboard_search_results = store.search(
                    embedding, top_k=5
                )
            except Exception as error:
                st.error(f"Could not search project information: {error}")
                st.session_state.dashboard_search_results = None

    search_results = st.session_state.dashboard_search_results
    if search_results is not None:
        with st.container(border=True):
            st.markdown("**Search results**")
            for result in search_results:
                st.markdown(
                    f"**{result['source']}** · Chunk {result['chunk_index'] + 1} "
                    f"· Similarity {result['similarity']:.0%}"
                )
                st.write(result["text"])

    risk_counts = _risk_counts(totals["risks"])
    doc_count = store.source_count()
    metrics = st.columns(3)
    metrics[0].metric(
        "Documents indexed",
        doc_count,
        help="Persistent document sources in the local knowledge base.",
        border=True,
    )
    metrics[1].metric(
        "Project health",
        f"{health['overall_score']}/100" if health else "Not analyzed",
        help="Rule-based score from the current analyzed document evidence.",
        border=True,
    )
    metrics[2].metric(
        "Identified risks",
        len(totals["risks"]) if analyses else "Not analyzed",
        border=True,
    )
    metrics = st.columns(2)
    metrics[0].metric(
        "Unresolved blockers",
        len(totals["blockers"]) if analyses else "Not analyzed",
        border=True,
    )
    metrics[1].metric(
        "Action items identified",
        len(totals["actions"]) if analyses else "Not analyzed",
        help="Detected action statements; completion status is not tracked.",
        border=True,
    )

    st.subheader("Scope summary")
    scope_column, delivery_column = st.columns(2)
    with scope_column:
        with st.container(border=True):
            st.markdown("**Goals and deliverables**")
            outcomes = [
                finding
                for finding in totals["findings"]
                if {"goal", "deliverable"} & set(finding["categories"])
            ]
            if outcomes:
                for finding in outcomes[:6]:
                    st.write(finding["text"])
                    st.caption(
                        f"{', '.join(finding['categories'])} · {finding['source']}"
                    )
            elif analyses:
                st.caption("No explicit goals or deliverables were identified.")
            else:
                st.caption("Analyze project documents to summarize scope.")
    with delivery_column:
        with st.container(border=True):
            st.markdown("**Milestones, timelines, and owners**")
            delivery_findings = [
                finding
                for finding in totals["findings"]
                if {"milestone", "timeline", "responsibility"}
                & set(finding["categories"])
            ]
            if delivery_findings:
                for finding in delivery_findings[:6]:
                    st.write(finding["text"])
                    details = [finding["source"]]
                    if finding.get("owner"):
                        details.append(f"Owner: {finding['owner']}")
                    if finding.get("timeline"):
                        details.append(f"Date: {finding['timeline']}")
                    st.caption(" · ".join(details))
            elif analyses:
                st.caption("No explicit milestones, dates, or owners were identified.")
            else:
                st.caption("Analyze project documents to summarize delivery commitments.")

    st.subheader("Project analytics")
    health_column, risk_column, milestone_column = st.columns([1, 1, 1.15])
    with health_column:
        with st.container(border=True):
            st.markdown("**Project health score**")
            if health:
                st.altair_chart(
                    _health_ring(score),
                    alt=f"Project health score is {score} out of 100",
                    width="stretch",
                )
                st.caption(
                    "Evidence-based health score · not a task completion estimate."
                )
            else:
                st.info("Analyze documents to calculate project health.")
    with risk_column:
        with st.container(border=True):
            st.markdown("**Risk distribution**")
            chart = _risk_donut(totals["risks"])
            if not analyses:
                st.caption("Analyze project documents to calculate risk distribution.")
            elif chart is not None:
                st.altair_chart(
                    chart,
                    alt="Distribution of explicit risks by agent severity",
                    width="stretch",
                )
                st.caption("The current agents classify explicit signals as high or medium.")
            elif analyses:
                st.caption("No explicit risk signals in analyzed documents.")
    with milestone_column:
        with st.container(border=True):
            st.markdown("**Milestones by source**")
            if totals["milestones"]:
                milestone_counts = {}
                for milestone in totals["milestones"]:
                    source = milestone["source"]
                    milestone_counts[source] = milestone_counts.get(source, 0) + 1
                milestone_data = [
                    {"source": source, "milestones": count}
                    for source, count in milestone_counts.items()
                ]
                chart = (
                    alt.Chart(alt.Data(values=milestone_data))
                    .mark_bar(color="#548ef0", cornerRadiusEnd=5)
                    .encode(
                        x=alt.X("milestones:Q", title="Signals"),
                        y=alt.Y("source:N", sort="-x", title=None),
                        tooltip=["source:N", "milestones:Q"],
                    )
                    .properties(height=170)
                    .configure_axis(
                        labelColor="#c5d3e9",
                        titleColor="#93a4c2",
                        gridColor="#243552",
                    )
                    .configure_view(stroke=None)
                )
                st.altair_chart(
                    chart,
                    alt="Count of milestone statements identified in each source document",
                    width="stretch",
                )
            else:
                st.caption(
                    "No milestone statements have been identified."
                    if analyses
                    else "Analyze documents to identify milestone statements."
                )

    details_column, action_column = st.columns([1.1, 1])
    with details_column:
        st.subheader("AI risk intelligence")
        if totals["risks"]:
            for risk in totals["risks"][:5]:
                with st.container(border=True):
                    st.badge(
                        risk["severity"].title(),
                        color="red" if risk["severity"] == "high" else "orange",
                    )
                    st.write(risk["evidence"])
                    st.caption(
                        f"Potential impact signal from source: {risk['evidence']}"
                    )
                    matches = [
                        action
                        for action in totals["actions"]
                        if action["source"] == risk["source"]
                    ]
                    if matches:
                        st.caption(f"Documented follow-up: {matches[0]['text']}")
                    else:
                        st.caption("Potential impact: see the source evidence above.")
                        st.caption("No mitigation action was explicitly documented.")
                    _source_caption(risk["source"])
        elif not analyses:
            st.info("Analyze project documents to identify risk signals.")
        else:
            st.info("No explicit delivery risk signals have been identified.")
        if analyses:
            st.caption(
                f"High {risk_counts['high']} · Medium {risk_counts['medium']}. "
                "The rule-based agents currently emit high and medium risk labels only."
            )

    with action_column:
        st.subheader("Project health overview")
        with st.container(border=True):
            if health:
                for key in ("scope_clarity", "timeline_risk", "blockers"):
                    dimension = health["dimensions"][key]
                    label = key.replace("_", " ").title()
                    st.markdown(f"**{label}** · {dimension['score']}/100")
                    st.progress(dimension["score"] / 100)
                    st.caption(dimension["rationale"])
                st.markdown(
                    f"**Health explanation:** {health['scoring_note']} "
                    f"{health['dimensions']['scope_clarity']['rationale']} "
                    f"{health['dimensions']['timeline_risk']['rationale']} "
                    f"{health['dimensions']['blockers']['rationale']}"
                )
            else:
                st.info("Analyze project documents to see health dimensions.")

    st.subheader("AI project recommendations")
    recommendations = _dashboard_recommendations(analyses, totals)
    if recommendations:
        st.info(
            f"Insight of the day: {recommendations[0]['text']}",
            icon=":material/lightbulb:",
        )
        for recommendation in recommendations:
            with st.container(border=True):
                st.badge(
                    recommendation["priority"],
                    color={
                        "High": "red",
                        "Medium": "orange",
                        "Low": "green",
                    }.get(recommendation["priority"], "blue"),
                )
                st.write(recommendation["text"])
                st.caption(recommendation["detail"])
                if recommendation["source"]:
                    _source_caption(recommendation["source"])
    else:
        st.caption("Run Project analysis agents to generate evidence-based recommendations.")

    activity_column, upcoming_column = st.columns(2)
    with activity_column:
        st.subheader("Recent project activity")
        events = st.session_state.recent_activity
        if events:
            with st.container(border=True):
                for event in events[:6]:
                    st.markdown(f"**{event['kind']}** · {event['detail']}")
                    suffix = f" · {event['source']}" if event["source"] else ""
                    st.caption(f"{event['timestamp']}{suffix}")
        else:
            st.caption(
                "Activity is recorded for this browser session as documents are "
                "uploaded, analyzed, or reports are generated."
            )
        recent_risks = totals["risks"][-3:]
        if recent_risks:
            st.markdown("**Latest identified risks**")
            for risk in reversed(recent_risks):
                st.markdown(f":orange[{risk['severity'].title()}] · {risk['evidence']}")
                _source_caption(risk["source"])
        st.caption("Activity history is session-only and is not retained after restart.")

    with upcoming_column:
        st.subheader("Upcoming tasks and deadlines")
        if totals["actions"]:
            with st.container(border=True):
                for action in totals["actions"][:6]:
                    st.markdown(f"**{action['text']}**")
                    details = []
                    if action.get("owner"):
                        details.append(f"Owner: {action['owner']}")
                    if action.get("due_date"):
                        details.append(f"Due: {action['due_date']}")
                    details.append("Status not tracked")
                    st.caption(" · ".join(details))
                    _source_caption(action["source"])
        else:
            st.caption(
                "No explicit action items were identified."
                if analyses
                else "Analyze documents to identify action items."
            )

    agent_column, trend_column = st.columns(2)
    with agent_column:
        st.subheader("AI agent activity")
        with st.container(border=True):
            for name in ("Scope agent", "Risk agent", "Blocker agent"):
                status = (
                    f"Analyzed {len(analyses)} document(s)"
                    if analyses
                    else "Ready · awaiting analysis"
                )
                st.markdown(f"**{name}** · {status}")
            documentation_status = (
                "Generated in this session"
                if st.session_state.generated_documentation
                else "Ready · not generated"
            )
            st.markdown(f"**Documentation agent** · {documentation_status}")
    with trend_column:
        st.subheader("Project health trend")
        with st.container(border=True):
            st.caption(
                "Historical health scores are not stored yet, so a trend cannot "
                "be displayed. The score above reflects the current analysis only."
            )

    st.subheader("Project timeline")
    dated_milestones = [
        milestone
        for milestone in totals["milestones"]
        if milestone.get("timeline")
    ]
    if dated_milestones:
        for milestone in dated_milestones[:8]:
            st.markdown(
                f"**{milestone['timeline']}** · {milestone['text']}"
            )
            _source_caption(milestone["source"])
    elif analyses:
        st.caption("No dated milestones are available in the analyzed documents.")
    else:
        st.caption("Analyze documents to identify project milestones.")


def _render_documents(store, uploaded_files):
    st.title("Documents & knowledge base")
    st.caption(
        "New or updated files are indexed and analyzed automatically. Existing "
        "knowledge-base documents remain available when you add progress updates."
    )

    st.subheader("Selected documents")
    if uploaded_files:
        st.write(
            f"{len(uploaded_files)} file(s) selected · Supported formats: "
            "PDF, DOCX, CSV, TXT"
        )
        for uploaded_file in uploaded_files:
            try:
                indexed = (
                    store.get_document_hash(uploaded_file.name)
                    == document_content_hash(uploaded_file)
                )
            except Exception:
                indexed = False
            state = "Indexed" if indexed else "Waiting to index"
            st.markdown(f"- `{uploaded_file.name}` · {state}")
    else:
        st.info(
            "Use the sidebar uploader to add PDF, DOCX, CSV, or TXT files. "
            "Previously indexed documents remain in the local knowledge base."
        )

    st.caption("Standard chunking: 300 words per chunk with 50 words of overlap.")

    st.divider()
    knowledge_column, chunks_column = st.columns(2)
    knowledge_column.metric("Documents indexed", store.source_count())
    chunks_column.metric("Searchable chunks", f"{store.chunk_count():,}")

    st.subheader("Search indexed documents")
    with st.form("document_search"):
        query = st.text_input(
            "Search your project documents",
            placeholder="Try: launch date, API dependency, or project goal",
        )
        top_k = st.slider(
            "Passages to retrieve", min_value=1, max_value=10, value=5
        )
        search_submitted = st.form_submit_button(
            "Search documents", type="primary"
        )

    if search_submitted:
        if not query.strip():
            st.warning("Enter a question or search phrase.")
        elif store.chunk_count() == 0:
            st.info("Upload and index documents before searching.")
        else:
            try:
                model = load_embedding_model()
                query_embedding = model.encode(
                    [query],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )[0]
                results = store.search(query_embedding, top_k=top_k)
            except Exception as error:
                st.error(f"Could not search the knowledge base: {error}")
            else:
                if not results:
                    st.info("No matching passages were found.")
                for result in results:
                    with st.container(border=True):
                        st.markdown(
                            f"**{result['source']}** · "
                            f"Chunk {result['chunk_index'] + 1} · "
                            f"Similarity {result['similarity']:.0%}"
                        )
                        st.write(result["text"])


def _render_project_analysis(analyses, uploaded_files, store):
    st.title("Project analysis")
    st.caption(
        "New uploads are analyzed automatically. Findings retain source evidence "
        "and are persisted with the local knowledge base."
    )
    if st.button(
        "Re-analyze selected uploads",
        type="primary",
        disabled=not uploaded_files,
    ):
        _ingest_uploads(uploaded_files, store, force=True)
        saved_analyses = store.list_document_analyses()
        if saved_analyses:
            st.session_state.project_analyses = saved_analyses
        analyses = st.session_state.project_analyses

    if not analyses:
        st.info(
            "Select project files in the sidebar and analyze them to see "
            "scope, risks, blockers, decisions, and actions."
        )
        return

    totals = _analysis_totals(analyses)
    counts = st.columns(5)
    counts[0].metric("Analyzed documents", len(analyses))
    counts[1].metric("Scope signals", len(totals["findings"]))
    counts[2].metric("Risks", len(totals["risks"]))
    counts[3].metric("Blockers", len(totals["blockers"]))
    counts[4].metric("Actions", len(totals["actions"]))

    for analysis in analyses:
        with st.expander(
            f"Document findings · {analysis['scope']['findings'][0]['source'] if analysis['scope']['findings'] else 'Source document'}",
            expanded=True,
        ):
            findings = analysis["scope"]["findings"]
            scope_findings = [
                finding
                for finding in findings
                if {"goal", "deliverable", "responsibility"}
                & set(finding["categories"])
            ]
            timeline_findings = [
                finding
                for finding in findings
                if {"milestone", "timeline"} & set(finding["categories"])
            ]
            risks = analysis["delivery_risk"]["risks"]
            items = analysis["blockers_and_actions"]

            tab_scope, tab_timeline, tab_risks, tab_blockers, tab_decisions, tab_actions = st.tabs(
                (
                    "Scope & deliverables",
                    "Milestones & timelines",
                    "Risks",
                    "Blockers",
                    "Decisions",
                    "Actions",
                )
            )
            with tab_scope:
                _render_items(
                    scope_findings,
                    "text",
                    "No explicit goals or deliverables were identified.",
                    lambda finding: [
                        ", ".join(finding["categories"]),
                        *(
                            [f"Owner: {finding['owner']}"]
                            if finding["owner"]
                            else []
                        ),
                    ],
                )
            with tab_timeline:
                _render_items(
                    timeline_findings,
                    "text",
                    "No explicit milestones or timelines were identified.",
                    lambda finding: (
                        [f"Timeline: {finding['timeline']}"]
                        if finding["timeline"]
                        else [", ".join(finding["categories"])]
                    ),
                )
            with tab_risks:
                _render_items(
                    risks,
                    "evidence",
                    "No explicit delivery risks were identified.",
                    lambda risk: [
                        f"Severity: {risk['severity'].title()}",
                        ", ".join(risk["signals"]),
                    ],
                )
                st.caption(analysis["delivery_risk"]["limitation"])
            with tab_blockers:
                _render_items(
                    items["blockers"],
                    "text",
                    "No explicit blockers were identified.",
                )
            with tab_decisions:
                _render_items(
                    items["pending_decisions"],
                    "text",
                    "No pending decisions were identified.",
                )
            with tab_actions:
                _render_items(
                    items["actions"],
                    "text",
                    "No explicit action items were identified.",
                    lambda action: [
                        *(
                            [f"Owner: {action['owner']}"]
                            if action.get("owner")
                            else []
                        ),
                        *(
                            [f"Due: {action['due_date']}"]
                            if action.get("due_date")
                            else []
                        ),
                    ],
                )


def _render_risk_analysis(analyses):
    st.title("Risk analysis")
    st.caption(
        "Review rule-based delivery risk signals with their source evidence. "
        "Severity is limited to the explicit labels produced by existing agents."
    )
    if not analyses:
        st.info("Run Project analysis agents on uploaded documents to populate the risk register.")
        return

    totals = _analysis_totals(analyses)
    counts = _risk_counts(totals["risks"])
    columns = st.columns(2)
    for column, severity in zip(columns, ("high", "medium")):
        column.metric(
            f"{severity.title()} signals",
            counts[severity],
            border=True,
        )

    if not totals["risks"]:
        st.success("No explicit delivery risk signals were identified.")
    for risk in totals["risks"]:
        with st.container(border=True):
            st.badge(
                risk["severity"].title(),
                color="red" if risk["severity"] == "high" else "orange",
            )
            st.write(risk["evidence"])
            related_actions = [
                action
                for action in totals["actions"]
                if action["source"] == risk["source"]
            ]
            if related_actions:
                st.markdown("**Documented follow-up**")
                st.write(related_actions[0]["text"])
            else:
                st.caption("No mitigation action was explicitly documented.")
            _source_caption(risk["source"])

    with st.container(border=True):
        st.markdown("**Risk heatmap**")
        st.caption(
            "Probability and impact are not estimated by the existing agents. "
            "A heatmap would imply measurements the source analysis does not provide."
        )


def _build_report_markdown(
    documentation, health, analyses=None, project_name=None
):
    sections = [
        f"# {project_name or 'Project documentation'}",
        "## Project summary",
        f"- Analyzed documents: {len(analyses or [])}",
        f"- Overall health: {health['overall_score']}/100",
        "## Scope summary",
    ]
    scope_findings = [
        finding
        for analysis in analyses or []
        for finding in analysis["scope"]["findings"]
    ]
    for finding in scope_findings:
        details = [", ".join(finding["categories"])]
        if finding.get("owner"):
            details.append(f"Owner: {finding['owner']}")
        if finding.get("timeline"):
            details.append(f"Date: {finding['timeline']}")
        sections.append(
            f"- {finding['text']} "
            f"({' · '.join(details)}; Source: {finding['source']})"
        )
    if not scope_findings:
        sections.append("No explicit scope or delivery commitments were identified.")

    sections.append("## User stories")
    for story in documentation["user_stories"]:
        sections.append(
            f"- {story['story']}  \n"
            f"  Evidence ({story['source']}): {story['evidence']}"
        )
    if not documentation["user_stories"]:
        sections.append("No explicit goals or deliverables were identified.")

    sections.append("## Risk register")
    for risk in documentation["risk_register"]:
        sections.append(
            f"- **{risk['severity'].title()}** — {risk['description']} "
            f"(Source: {risk['source']})"
        )
    if not documentation["risk_register"]:
        sections.append("No explicit risk signals were identified.")

    sections.append("## Action items")
    for action in documentation["actions"]:
        details = []
        if action.get("owner"):
            details.append(f"Owner: {action['owner']}")
        if action.get("due_date"):
            details.append(f"Due: {action['due_date']}")
        suffix = f" ({'; '.join(details)})" if details else ""
        sections.append(
            f"- {action['text']}{suffix} (Source: {action['source']})"
        )
    if not documentation["actions"]:
        sections.append("No explicit action items were identified.")

    sections.append("## Blockers")
    blockers = [
        blocker
        for analysis in analyses or []
        for blocker in analysis["blockers_and_actions"]["blockers"]
    ]
    for blocker in blockers:
        sections.append(f"- {blocker['text']} (Source: {blocker['source']})")
    if not blockers:
        sections.append("No explicit blockers were identified.")

    sections.append("## Project recommendations")
    recommendations = _dashboard_recommendations(
        analyses or [], _analysis_totals(analyses or [])
    )
    for recommendation in recommendations:
        source = (
            f" (Source: {recommendation['source']})"
            if recommendation["source"]
            else ""
        )
        sections.append(
            f"- **{recommendation['priority']}** — "
            f"{recommendation['text']} {recommendation['detail']}{source}"
        )
    if not recommendations:
        sections.append("No recommendations were generated.")

    sections.append("## Health scoring")
    for label, dimension in health["dimensions"].items():
        sections.append(
            f"- **{label.replace('_', ' ').title()}:** "
            f"{dimension['score']}/100 — {dimension['rationale']}"
        )
    sections.append(health["scoring_note"])
    return "\n\n".join(sections)


def _render_health_and_reports(analyses):
    st.title("Project health")
    st.caption(
        "Review transparent health indicators calculated from explicit document signals."
    )
    if not analyses:
        st.info(
            "Run Project analysis agents on your uploaded documents to calculate health."
        )
        return

    health = calculate_project_health(analyses)
    totals = _analysis_totals(analyses)
    st.subheader("Overall project health")
    score_column, scope_column, timeline_column, blocker_column = st.columns(4)
    score_column.metric("Overall score", f"{health['overall_score']}/100")
    scope_column.metric(
        "Scope clarity", f"{health['dimensions']['scope_clarity']['score']}/100"
    )
    timeline_column.metric(
        "Timeline risk",
        f"{health['dimensions']['timeline_risk']['score']}/100",
    )
    blocker_column.metric(
        "Blocker status",
        f"{health['blocker_count']} identified",
        delta="Needs attention" if health["blocker_count"] else "Clear",
        delta_color="inverse" if health["blocker_count"] else "normal",
    )

    st.markdown("### Health dimensions")
    for label, dimension in health["dimensions"].items():
        st.markdown(f"**{label.replace('_', ' ').title()}**")
        st.progress(dimension["score"] / 100)
        st.caption(
            f"{dimension['score']}/100 · {dimension['rationale']}"
        )
    st.caption(
        f"{totals['high_risks']} high and {totals['medium_risks']} medium "
        f"explicit risk signals. {health['scoring_note']}"
    )


def _render_documentation_generator(analyses):
    st.title("Documentation generator")
    st.caption(
        "Generate an evidence-linked project report from analyzed documents."
    )
    if not analyses:
        st.info("Run Project analysis agents before generating documentation.")
        return

    health = calculate_project_health(analyses)
    st.divider()
    if st.button("Generate project report", type="primary"):
        try:
            documentation = generate_project_documentation(analyses)
            report = _build_report_markdown(
                documentation,
                health,
                analyses,
                st.session_state.project_name,
            )
        except Exception as error:
            st.error(f"Could not generate the project report: {error}")
        else:
            st.session_state.generated_documentation = documentation
            st.session_state.generated_report = report
            _record_activity(
                "Documentation generated",
                "Evidence-linked project report created",
            )
            st.success("Evidence-linked project report generated.")

    documentation = st.session_state.generated_documentation
    report = st.session_state.generated_report
    if documentation and report:
        tabs = st.tabs(("User stories", "Risk register", "Action items"))
        with tabs[0]:
            _render_items(
                documentation["user_stories"],
                "story",
                "No explicit goals or deliverables were identified.",
                lambda story: [f"Evidence: {story['evidence']}"],
            )
        with tabs[1]:
            _render_items(
                documentation["risk_register"],
                "description",
                "No explicit risk signals were identified.",
                lambda risk: [
                    f"Severity: {risk['severity'].title()}",
                    f"Evidence: {risk['evidence']}",
                ],
            )
        with tabs[2]:
            _render_items(
                documentation["actions"],
                "text",
                "No explicit action items were identified.",
                lambda action: [
                    *(
                        [f"Owner: {action['owner']}"]
                        if action.get("owner")
                        else []
                    ),
                    *(
                        [f"Due: {action['due_date']}"]
                        if action.get("due_date")
                        else []
                    ),
                ],
            )
        st.download_button(
            "Download project report",
            data=report,
            file_name="project-documentation.md",
            mime="text/markdown",
            type="primary",
        )
        with st.expander("Preview full report"):
            st.markdown(report)


def _queue_chat_question(question):
    st.session_state.pending_chat_question = question


def _render_assistant(store):
    st.title("AI project assistant")
    st.caption(
        "Ask questions about indexed documents. Answers quote retrieved "
        "passages and cite their source."
    )
    if store.chunk_count() == 0:
        st.info(
            "Build the knowledge base in Documents before asking questions."
        )
        return

    top_row, clear_column = st.columns([5, 1])
    with top_row:
        st.markdown("### Suggested questions")
    with clear_column:
        if st.button("Clear chat"):
            st.session_state.project_chat_history = []
            st.rerun()

    suggestions = st.columns(4)
    for column, question in zip(suggestions, SUGGESTED_QUESTIONS):
        column.button(
            question,
            key=f"suggest_{question}",
            on_click=_queue_chat_question,
            args=(question,),
        )

    for message in st.session_state.project_chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                st.write(message["answer"])
                for evidence in message["evidence"]:
                    st.markdown(
                        f"**{evidence['source']}** · "
                        f"Chunk {evidence['chunk_index'] + 1} · "
                        f"Similarity {evidence['similarity']:.0%}"
                    )
                    st.write(f"> {evidence['text']}")

    chat_question = st.chat_input(
        "Ask about project scope, progress, or risks",
        key="project_chat_input",
    )
    pending_question = st.session_state.pending_chat_question
    st.session_state.pending_chat_question = None
    question = pending_question or chat_question
    if question:
        st.session_state.project_chat_history.append(
            {"role": "user", "content": question}
        )
        try:
            model = load_embedding_model()
            query_embedding = model.encode(
                [question],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            retrieved = store.search(query_embedding, top_k=5)
            response = answer_project_question(question, retrieved)
        except Exception as error:
            st.error(f"Could not answer the project question: {error}")
        else:
            assistant_message = {
                "role": "assistant",
                "answer": response["answer"],
                "evidence": response["evidence"],
            }
            st.session_state.project_chat_history.append(assistant_message)
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(response["answer"])
                if response["evidence"]:
                    st.markdown("**Sources and retrieved evidence**")
                for evidence in response["evidence"]:
                    st.markdown(
                        f"**{evidence['source']}** · "
                        f"Chunk {evidence['chunk_index'] + 1} · "
                        f"Similarity {evidence['similarity']:.0%}"
                    )
                    st.write(f"> {evidence['text']}")


def _render_settings(store):
    st.title("Settings")
    st.caption("Manage the project workspace and review local data status.")
    st.text_input("Project name", key="project_name")
    with st.container(border=True):
        st.markdown("**Local knowledge base**")
        st.write(f"Indexed documents: {store.source_count()}")
        st.write(f"Searchable chunks: {store.chunk_count():,}")
        st.caption(
            "Source evidence and embeddings are stored in the local SQLite index."
        )
    with st.container(border=True):
        st.markdown("**Analysis methodology**")
        st.write(
            "Project agents use transparent text rules and retain matched source "
            "evidence. The current health score is a rule-based indicator, not a "
            "prediction of project success."
        )
        st.caption(
            "Task completion, risk probability/impact, and historical health trends "
            "are not currently persisted or measured."
        )


def run_app():
    st.set_page_config(
        page_title="AI Project Intelligence & Risk Advisor",
        page_icon=":material/dashboard:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _initialize_session_state()
    if st.session_state.get("active_page") not in NAVIGATION:
        st.session_state.active_page = NAVIGATION[0]
    _apply_styles()

    with st.sidebar:
        st.markdown("## ◈ Project intelligence")
        st.caption("AI-powered delivery workspace")
        st.divider()
        selected_page = st.radio(
            "Workspace navigation",
            NAVIGATION,
            format_func=lambda value: value.split("  ", 1)[-1],
            key="active_page",
            label_visibility="collapsed",
        )
        st.divider()
        uploaded_files = st.file_uploader(
            "Add project documents",
            type=["pdf", "docx", "csv", "txt"],
            accept_multiple_files=True,
            key="project_document_upload",
            help=(
                "New and changed files are indexed and analyzed automatically. "
                "Supported formats: PDF, DOCX, CSV, and TXT."
            ),
        )
        st.caption("PDF · DOCX · CSV · TXT")
        store = load_vector_store()
        st.divider()
        st.caption("Source evidence stays on this machine.")

    saved_analyses = store.list_document_analyses()
    if saved_analyses:
        st.session_state.project_analyses = saved_analyses
    _ingest_uploads(uploaded_files, store)
    with st.sidebar:
        st.metric("Indexed documents", store.source_count())
        st.metric("Knowledge base chunks", f"{store.chunk_count():,}")
    _sync_upload_activity(uploaded_files)
    analyses = st.session_state.project_analyses
    project_name = st.session_state.project_name.strip() or "Project workspace"
    page = PAGE_LABELS[selected_page]

    if page == "Dashboard":
        _render_dashboard(analyses, store, project_name)
    elif page == "Document intelligence":
        _render_documents(store, uploaded_files)
    elif page == "Project analysis agents":
        _render_project_analysis(analyses, uploaded_files, store)
    elif page == "Risk analysis":
        _render_risk_analysis(analyses)
    elif page == "Project health":
        _render_health_and_reports(analyses)
    elif page == "Documentation generator":
        _render_documentation_generator(analyses)
    elif page == "AI project assistant":
        _render_assistant(store)
    elif page == "Settings":
        _render_settings(store)

    st.divider()
    st.caption(
        "Project signals are rule-based and grounded in uploaded text. "
        "Health scores are indicators, not predictions of project success."
    )
