"""Student performance dashboard. Run with: streamlit run app.py"""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import TARGET, clean, load_data


st.set_page_config(
    page_title="Student Performance | Insight",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#15243b; --muted:#718096; --blue:#365cf5; --line:#e8edf5; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
    [data-testid="stAppViewContainer"] { background:#f5f7fb; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#101b31; }
    [data-testid="stSidebar"] * { color:#e8eefb; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#b9c5dc; }
    [data-testid="stSidebar"] input { background:#fff !important; color:#15243b !important; }
    .block-container { max-width:1440px; padding-top:2rem; padding-bottom:3rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif; color:var(--ink); letter-spacing:-.035em; }
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 { color:var(--ink) !important; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCaptionContainer"] { color:#526176 !important; }
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] li p,
    [data-testid="stMarkdownContainer"] strong { color:#334155 !important; }
    [data-testid="stTabs"] button { color:#526176 !important; }
    [data-testid="stTabs"] button[aria-selected="true"] { color:#365cf5 !important; }
    [data-testid="stForm"] label, [data-testid="stForm"] label p { color:#27364d !important; }
    [data-testid="stForm"] [data-testid="stMarkdownContainer"] p { color:#526176 !important; }
    [data-testid="stFormSubmitButton"] button,
    div.stButton > button { background:#365cf5 !important; color:#fff !important; }
    .hero { background:linear-gradient(115deg,#142342 0%,#253e77 58%,#365cf5 100%); border-radius:22px;
        padding:30px 34px; color:white; margin:4px 0 24px; box-shadow:0 14px 32px rgba(25,48,99,.15); }
    .hero h1 { color:white !important; font-size:2rem; margin:4px 0 7px; }
    .hero p { color:#d9e2fb !important; margin:0; font-size:1rem; }
    .eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; color:#a9beff; }
    .section-label { color:#596a83; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; }
    [data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:16px; padding:18px 20px; box-shadow:0 4px 15px rgba(29,51,88,.035); }
    [data-testid="stMetricLabel"] { color:#718096; }
    [data-testid="stMetricValue"] { color:var(--ink); font-family:'Manrope',sans-serif; font-size:1.25rem !important; }
    div[data-testid="stTabs"] button { font-weight:600; }
    div[data-testid="stForm"] { background:white; padding:22px 24px 18px; border:1px solid var(--line); border-radius:18px; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button { border-radius:10px; border:0; background:#365cf5; color:white; font-weight:700; padding:.65rem 1.2rem; }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover { background:#2549dd !important; color:white !important; }
    .result-card { background:white; border:1px solid var(--line); border-radius:18px; padding:24px; box-shadow:0 4px 15px rgba(29,51,88,.035); }
    .small-note { color:#718096; font-size:.86rem; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div style='font-size:1.45rem;font-weight:800;font-family:Manrope'>✦ Student Insight</div>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='margin-top:3px'>Performance analytics workspace</p>", unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.markdown("<div class='section-label'>Workspace</div>", unsafe_allow_html=True)
data_path = st.sidebar.text_input("Dataset file", "data/student-mat.csv", help="CSV file from the UCI Student Performance dataset")
st.sidebar.caption("The selected data file is used to populate insights and prediction defaults.")

model_path = "models/best_model.joblib"
if not os.path.isfile(data_path):
    st.markdown("<div class='hero'><div class='eyebrow'>Student analytics</div><h1>Welcome to Student Insight</h1><p>Explore academic patterns and estimate final grades from student progress.</p></div>", unsafe_allow_html=True)
    st.error(f"Dataset file not found: `{data_path}`")
    st.info("Use `data/student-mat.csv` or enter `data/student-por.csv` in the sidebar.")
    st.stop()

df = clean(load_data(data_path))

st.markdown(
    "<div class='hero'><div class='eyebrow'>Student analytics · Overview</div>"
    "<h1>Understand performance. Plan support.</h1>"
    "<p>Explore the dataset, compare model quality, and estimate a student's final grade.</p></div>",
    unsafe_allow_html=True,
)

if not os.path.isfile(model_path):
    st.warning("The prediction model has not been trained yet. Run `python pipeline.py` in the project terminal, then refresh this page.")
    st.stop()

bundle = joblib.load(model_path)
if TARGET not in df.columns:
    st.error(f"The selected CSV must include a `{TARGET}` final-grade column.")
    st.stop()

avg_grade = float(df[TARGET].mean())
pass_rate = float((df[TARGET] >= 10).mean() * 100)
top_grade = float((df[TARGET] >= 16).mean() * 100)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Students in dataset", f"{len(df):,}", help="Rows remaining after duplicate and missing-target cleanup")
k2.metric("Average final grade", f"{avg_grade:.1f} / 20")
k3.metric("At or above 10 / 20", f"{pass_rate:.0f}%")
k4.metric("Best model", bundle["best"], help=f"Full model name: {bundle['best']}")

st.write("")
predict_tab, insights_tab, models_tab = st.tabs(["✦  Grade prediction", "▥  Dataset insights", "⌁  Model performance"])

with predict_tab:
    left, right = st.columns([1.35, .85], gap="large")
    with left:
        st.subheader("Student profile")
        st.markdown("<p class='small-note'>Choose the profile this estimate is for. The dataset has anonymous records, so you can add your own label or ID.</p>", unsafe_allow_html=True)
        with st.form("prediction_form"):
            profile_mode = st.radio("Profile starting point", ["New student (typical dataset profile)", "Existing anonymous dataset record"], horizontal=True)
            record_number = None
            if profile_mode == "Existing anonymous dataset record":
                record_number = st.selectbox("Choose a dataset student record", range(1, len(df) + 1), format_func=lambda n: f"Student record #{n:03d}")
                base_row = df.iloc[record_number - 1].to_dict()
                default_label = f"Student record #{record_number:03d}"
            else:
                base_row = dict(bundle["defaults"])
                for col in bundle["cats"]:
                    if col in df.columns and not df[col].dropna().empty:
                        base_row[col] = df[col].mode().iloc[0]
                default_label = "New student"
            student_label = st.text_input("Student name, initials, or ID (optional)", value=default_label, help="This label is only used to identify the result on this screen.")
            st.markdown("**Academic progress and attendance**")
            a, b, c = st.columns(3)
            studytime_default = int(np.clip(base_row.get("studytime", 2), 1, 4))
            absences_default = int(np.clip(base_row.get("absences", 4), 0, 75))
            failures_default = int(np.clip(base_row.get("failures", 0), 0, 3))
            g1_default = int(np.clip(base_row.get("G1", 10), 0, 20))
            g2_default = int(np.clip(base_row.get("G2", 10), 0, 20))
            studytime = a.select_slider("Weekly study time", options=[1, 2, 3, 4], value=studytime_default, format_func=lambda x: {1:"< 2 hours", 2:"2–5 hours", 3:"5–10 hours", 4:"> 10 hours"}[x])
            absences = b.number_input("Absences", min_value=0, max_value=75, value=absences_default, step=1)
            failures = c.selectbox("Past class failures", options=[0, 1, 2, 3], index=failures_default)
            d, e = st.columns(2)
            g1 = d.slider("Previous grade · G1", min_value=0, max_value=20, value=g1_default)
            g2 = e.slider("Previous grade · G2", min_value=0, max_value=20, value=g2_default)
            submitted = st.form_submit_button("Estimate final grade", use_container_width=True)

        if submitted:
            row = dict(base_row)
            row.update(studytime=studytime, absences=absences, failures=failures, G1=g1, G2=g2)
            try:
                prediction = float(np.clip(bundle["pipeline"].predict(pd.DataFrame([row])[bundle["features"]])[0], 0, 20))
                st.session_state["last_prediction"] = {
                    "grade": prediction,
                    "label": student_label.strip() or (f"Student record #{record_number:03d}" if record_number else "New student"),
                    "record_number": record_number,
                    "profile_mode": profile_mode,
                    "studytime": studytime,
                    "absences": absences,
                    "failures": failures,
                    "G1": g1,
                    "G2": g2,
                }
            except Exception as exc:
                st.error(f"Could not generate a prediction: {exc}")

    with right:
        st.subheader("Prediction summary")
        if "last_prediction" in st.session_state:
            result = st.session_state["last_prediction"]
            prediction = result["grade"]
            st.markdown(f"**Estimate for: {result['label']}**")
            with st.container(border=True):
                st.markdown("<div class='section-label'>Estimated final grade · G3</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font:800 3.5rem Manrope;color:#15243b;line-height:1.2;margin:8px 0'>{prediction:.1f}<span style='font-size:1.1rem;color:#718096'> / 20</span></div>", unsafe_allow_html=True)
                st.progress(prediction / 20)
                if prediction < 10:
                    st.error("Below the 10/20 reference mark. Consider checking in and offering additional support.")
                elif prediction < 14:
                    st.warning("Around the passing range. Regular progress check-ins may help.")
                else:
                    st.success("Positive estimate. Encourage the student to maintain their progress.")
                st.caption("This is a model estimate, not an official grade.")
            st.markdown("**Inputs used**")
            studytime_text = {1: "< 2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: "> 10 hours"}[result["studytime"]]
            st.markdown(
                f"- Weekly study time: **{studytime_text}**  \n"
                f"- Absences: **{result['absences']}** · Past failures: **{result['failures']}**  \n"
                f"- Previous grades: **G1 {result['G1']} / 20**, **G2 {result['G2']} / 20**"
            )
            if result["record_number"] is None:
                st.caption("New student profile: unedited fields use typical dataset values (numeric medians and most common categories).")
            else:
                st.caption(f"Based on anonymous dataset record #{result['record_number']:03d}; its G3 grade is not used as an input. Other profile fields come from that record.")
        else:
            st.info("Complete the profile and submit it. The estimate will be shown here with the exact student label and inputs.")
            st.markdown("<div class='result-card'><div class='section-label'>How the estimate is made</div><p class='small-note'>The model predicts the final grade (G3, on a 0–20 scale) from previous grades, study time, absences, past failures, and other profile fields. Existing dataset records are anonymous.</p></div>", unsafe_allow_html=True)
        st.caption(f"Current model: {bundle['best']}")

with insights_tab:
    st.subheader("Dataset insights")
    st.markdown("<p class='small-note'>A quick view of grade outcomes and study patterns in the selected CSV.</p>", unsafe_allow_html=True)
    chart1, chart2 = st.columns(2, gap="large")
    with chart1:
        st.markdown("**Final grade distribution**")
        st.bar_chart(df[TARGET].value_counts().sort_index(), color="#5573f7", height=300)
    with chart2:
        st.markdown("**Average grade by weekly study time**")
        study_summary = df.groupby("studytime")[TARGET].mean().rename("Average G3")
        st.bar_chart(study_summary, color="#37a88b", height=300)
    st.markdown("**Student records**")
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)
    st.caption(f"Showing up to 50 of {len(df):,} cleaned rows.")

with models_tab:
    st.subheader("Model performance")
    st.markdown("<p class='small-note'>Lower RMSE means predictions are closer to observed final grades. R² summarizes variance explained on the held-out split.</p>", unsafe_allow_html=True)
    results_path = "outputs/model_comparison.csv"
    if os.path.isfile(results_path):
        results = pd.read_csv(results_path).sort_values("RMSE")
        st.dataframe(results, use_container_width=True, hide_index=True)
        if "RMSE" in results.columns:
            st.markdown("**RMSE comparison**")
            st.bar_chart(results.set_index("Model")[["RMSE"]], color="#5573f7", height=300)
        if os.path.isfile("outputs/actual_vs_predicted.png"):
            st.image("outputs/actual_vs_predicted.png", caption="Actual and predicted grades on the holdout split", width=540)
    else:
        st.info("Model comparison results will appear after the training pipeline completes.")
    if os.path.isfile("outputs/eda_overview.png"):
        with st.expander("View training data charts"):
            st.image("outputs/eda_overview.png", caption="Exploratory data overview", use_container_width=True)
            if os.path.isfile("outputs/correlation_heatmap.png"):
                st.image("outputs/correlation_heatmap.png", caption="Feature correlation heatmap", use_container_width=True)

st.markdown("<div style='text-align:center;color:#8b98ad;font-size:.8rem;padding:28px 0 0'>STUDENT INSIGHT &nbsp;·&nbsp; A learning project for educational exploration</div>", unsafe_allow_html=True)
