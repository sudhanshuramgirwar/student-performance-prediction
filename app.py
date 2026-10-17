"""Student performance dashboard. Run with: streamlit run app.py"""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import TARGET, clean, load_data


st.set_page_config(
    page_title="Student Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#172b45; --muted:#6b7a8d; --accent:#27627a; --line:#e2e8f0; }
    html, body, [class*="css"] { font-family:'Segoe UI', Arial, sans-serif; }
    [data-testid="stAppViewContainer"] { background:#f4f7fb; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#13243b; border-right:1px solid #213851; }
    [data-testid="stSidebar"] * { color:#edf2f8; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#b9c6d6 !important; }
    [data-testid="stSidebar"] input { background:#fff !important; color:#203247 !important; }
    .block-container { max-width:1400px; padding-top:1.7rem; padding-bottom:2.5rem; }
    h1,h2,h3 { font-family:'Segoe UI', Arial, sans-serif; color:var(--ink); letter-spacing:-.02em; }
    h1 { font-size:2rem; }
    h2 { font-size:1.45rem; }
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 { color:var(--ink) !important; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCaptionContainer"] { color:#4f6072 !important; }
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] li p,
    [data-testid="stMarkdownContainer"] strong { color:#26384d !important; }
    [data-testid="stTabs"] button { color:#536579 !important; }
    [data-testid="stTabs"] button[aria-selected="true"] { color:var(--accent) !important; }
    [data-testid="stForm"] label, [data-testid="stForm"] label p { color:#27394d !important; }
    [data-testid="stForm"] [data-testid="stMarkdownContainer"] p { color:#43566a !important; }
    [data-testid="stFormSubmitButton"] button,
    div.stButton > button { background:var(--accent) !important; color:#fff !important; border:1px solid var(--accent) !important; }
    [data-testid="stFormSubmitButton"] button *,
    div.stButton > button * { color:#fff !important; }
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] button [data-testid="stMarkdownContainer"] p,
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] button span { color:#fff !important; font-weight:700 !important; }
    [data-testid="stForm"] [data-testid="stWidgetLabel"] p,
    [data-testid="stForm"] label p { color:#172b45 !important; font-weight:600; }
    [data-testid="stForm"] [data-testid="stCaptionContainer"] { color:#42546a !important; }
    .page-header { background:#183650; border-left:5px solid #4d9295; border-radius:10px;
        padding:24px 28px; margin:2px 0 22px; box-shadow:0 5px 14px rgba(25,48,72,.08); }
    .page-header h1 { color:#fff !important; font-size:2rem; margin:5px 0 7px; }
    .page-header p { color:#e0e9f1 !important; margin:0; font-size:1rem; }
    .eyebrow { text-transform:uppercase; letter-spacing:.11em; font-size:.7rem; font-weight:700; color:#a9d1d2; }
    .section-label { color:#53677c; text-transform:uppercase; letter-spacing:.09em; font-size:.7rem; font-weight:700; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px 18px; box-shadow:0 3px 12px rgba(24,46,74,.035); }
    [data-testid="stMetricLabel"] { color:#617184; }
    [data-testid="stMetricValue"] { color:var(--ink); font-family:'Segoe UI', Arial, sans-serif; font-size:1.5rem !important; }
    div[data-testid="stTabs"] button { font-weight:600; }
    div[data-testid="stForm"] { background:#fff; padding:22px 24px 18px; border:1px solid var(--line); border-radius:12px; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button { border-radius:8px; font-weight:600; padding:.62rem 1rem; }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover { background:#1c4d61 !important; border-color:#1c4d61 !important; color:#fff !important; }
    .small-note { color:var(--muted); font-size:.9rem; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div style='font-size:1.25rem;font-weight:700'>Student Performance</div>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='margin-top:3px'>Prediction and analysis</p>", unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.markdown("<div class='section-label'>Data source</div>", unsafe_allow_html=True)
data_path = st.sidebar.text_input("Dataset file", "data/student-mat.csv", help="CSV file from the UCI Student Performance dataset")
st.sidebar.caption("The selected data file is used to populate insights and prediction defaults.")

model_path = "models/best_model.joblib"
if not os.path.isfile(data_path):
    st.markdown("<div class='page-header'><div class='eyebrow'>Academic analytics</div><h1>Student Performance Prediction System</h1><p>Identify students who may need support using study time, attendance, and previous grades.</p></div>", unsafe_allow_html=True)
    st.error(f"Dataset file not found: `{data_path}`")
    st.info("Use `data/student-mat.csv` or enter `data/student-por.csv` in the sidebar.")
    st.stop()

raw_df = load_data(data_path)
if TARGET not in raw_df.columns:
    st.error(f"The selected CSV must include a `{TARGET}` final-grade column.")
    st.stop()
raw_rows = len(raw_df)
duplicate_rows = int(raw_df.duplicated().sum())
df = clean(raw_df)
if TARGET not in df.columns:
    st.error(f"The selected CSV must include a `{TARGET}` final-grade column.")
    st.stop()
missing_target_rows = int(raw_df[TARGET].isna().sum()) if TARGET in raw_df.columns else 0
remaining_feature_missing = int(df.drop(columns=[TARGET]).isna().sum().sum())
numeric_feature_count = len(df.drop(columns=[TARGET]).select_dtypes(include="number").columns)
categorical_feature_count = len(df.drop(columns=[TARGET]).select_dtypes(exclude="number").columns)

st.markdown(
    "<div class='page-header'><div class='eyebrow'>Academic analytics</div>"
    "<h1>Student Performance Prediction System</h1>"
    "<p>Estimate a student's final grade from study time, attendance, and previous grades, then review the model and dataset.</p></div>",
    unsafe_allow_html=True,
)

if not os.path.isfile(model_path):
    st.warning("The prediction model has not been trained yet. Run `python pipeline.py` in the project terminal, then refresh this page.")
    st.stop()

bundle = joblib.load(model_path)

avg_grade = float(df[TARGET].mean())
pass_rate = float((df[TARGET] >= 10).mean() * 100)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Student records", f"{len(df):,}", help="Rows remaining after duplicate and missing-target cleanup")
k2.metric("Average final grade", f"{avg_grade:.1f} / 20")
k3.metric("Pass rate (G3 ≥ 10)", f"{pass_rate:.0f}%")
k4.metric("Best model", bundle["best"], help=f"Full model name: {bundle['best']}")

st.sidebar.markdown("<div class='section-label'>Project sections</div>", unsafe_allow_html=True)
selected_section = st.sidebar.radio(
    "Project sections",
    ["Overview", "Dataset analysis", "Model comparison"],
    label_visibility="collapsed",
    help="Overview opens the student grade prediction form.",
)

if selected_section == "Overview":
    left, right = st.columns([1.35, .85], gap="large")
    with left:
        st.subheader("Student profile")
        st.markdown("<p class='small-note'>Enter the student's name or ID, then adjust the academic details to estimate a final grade.</p>", unsafe_allow_html=True)
        with st.form("prediction_form"):
            base_row = dict(bundle["defaults"])
            for col in bundle["cats"]:
                if col in df.columns and not df[col].dropna().empty:
                    base_row[col] = df[col].mode().iloc[0]
            student_label = st.text_input("Student name, initials, or ID (optional)", value="New student", help="This label is only used to identify the result on this screen.")
            st.markdown("**Academic progress and attendance**")
            a, b, c = st.columns([1.15, 1, 1.15], gap="medium")
            studytime_band_default = int(np.clip(base_row.get("studytime", 2), 1, 4))
            studytime_hours_default = {1: 1, 2: 3, 3: 6, 4: 8}[studytime_band_default]
            absences_default = int(np.clip(base_row.get("absences", 4), 0, 75))
            failures_default = int(np.clip(base_row.get("failures", 0), 0, 3))
            g1_default = int(np.clip(base_row.get("G1", 10), 0, 20))
            g2_default = int(np.clip(base_row.get("G2", 10), 0, 20))
            studytime_hours = a.slider("Study time (0–8 hours/week)", min_value=0, max_value=8, value=studytime_hours_default, step=1)
            absences = b.number_input("Absences", min_value=0, max_value=75, value=absences_default, step=1)
            failures = c.selectbox("Past class failures", options=[0, 1, 2, 3], index=failures_default)
            d, e = st.columns(2)
            g1 = d.slider("Previous grade · G1", min_value=0, max_value=20, value=g1_default)
            g2 = e.slider("Previous grade · G2", min_value=0, max_value=20, value=g2_default)
            submitted = st.form_submit_button("Estimate final grade", use_container_width=True)

        if submitted:
            row = dict(base_row)
            # UCI encodes studytime in four bands. The requested 0–8 hour input
            # is mapped to the matching first three bands before prediction.
            studytime = 1 if studytime_hours < 2 else 2 if studytime_hours < 5 else 3
            row.update(studytime=studytime, absences=absences, failures=failures, G1=g1, G2=g2)
            try:
                prediction = float(np.clip(bundle["pipeline"].predict(pd.DataFrame([row])[bundle["features"]])[0], 0, 20))
                st.session_state["last_prediction"] = {
                    "grade": prediction,
                    "label": student_label.strip() or "New student",
                    "studytime_hours": studytime_hours,
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
            st.markdown(
                f"- Study time per week: **{result['studytime_hours']} hours**  \n"
                f"- Absences: **{result['absences']}** · Past failures: **{result['failures']}**  \n"
                f"- Previous grades: **G1 {result['G1']} / 20**, **G2 {result['G2']} / 20**"
            )
            st.caption("Unedited fields use typical dataset values (numeric medians and most common categories).")
        else:
            st.info("Complete the profile and submit it. The estimate will be shown here with the exact student label and inputs.")
            st.markdown("<div class='result-card'><div class='section-label'>How the estimate is made</div><p class='small-note'>The model predicts the final grade (G3, on a 0–20 scale) from previous grades, study time, absences, past failures, and other profile fields.</p></div>", unsafe_allow_html=True)
        st.caption(f"Current model: {bundle['best']}")

if selected_section == "Dataset analysis":
    st.subheader("Dataset analysis")
    st.markdown("<p class='small-note'>Review the cleaned student records, grade patterns, and preprocessing steps used for the regression models.</p>", unsafe_allow_html=True)
    st.markdown("**Data preparation summary**")
    prep1, prep2, prep3, prep4 = st.columns(4)
    prep1.metric("Rows loaded", f"{raw_rows:,}")
    prep2.metric("Duplicate rows removed", f"{duplicate_rows:,}")
    prep3.metric("Rows missing final grade", f"{missing_target_rows:,}")
    prep4.metric("Predictor fields", f"{numeric_feature_count + categorical_feature_count}")
    st.caption(
        f"Cleaned dataset: {len(df):,} rows · {numeric_feature_count} numeric fields · "
        f"{categorical_feature_count} categorical fields · {remaining_feature_missing:,} predictor values need imputation."
    )
    with st.expander("Preprocessing used in model training"):
        st.markdown(
            "- Duplicate rows are removed; rows without a final grade (G3) are excluded.\n"
            "- Missing numeric predictor values are filled with the training-set median, then standardized.\n"
            "- Missing categorical predictor values are filled with the most common value, then one-hot encoded.\n"
            "- Unseen categories are safely ignored during prediction."
        )
    chart1, chart2 = st.columns(2, gap="large")
    with chart1:
        st.markdown("**Final grade distribution**")
        st.bar_chart(df[TARGET].value_counts().sort_index(), color="#27627a", height=300)
    with chart2:
        st.markdown("**Average grade by weekly study time**")
        study_summary = df.groupby("studytime")[TARGET].mean().rename("Average G3")
        st.bar_chart(study_summary, color="#7da7a3", height=300)
    st.markdown("**Student records**")
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)
    st.caption(f"Showing up to 50 of {len(df):,} cleaned rows.")

if selected_section == "Model comparison":
    st.subheader("Model comparison")
    st.markdown("<p class='small-note'>RMSE measures average prediction error in grade points (lower is better). R² shows how much variation in final grades the model explains (closer to 1 is better).</p>", unsafe_allow_html=True)
    results_path = "outputs/model_comparison.csv"
    if os.path.isfile(results_path):
        results = pd.read_csv(results_path).sort_values("RMSE")
        best_row = results.iloc[0]
        score1, score2, score3 = st.columns(3)
        score1.metric("Selected model", str(best_row["Model"]))
        score2.metric("Holdout RMSE", f"{best_row['RMSE']:.2f} grade points")
        score3.metric("Holdout R²", f"{best_row['R2']:.3f}")
        display_results = results.rename(columns={"R2": "R²", "CV_R2": "5-fold CV R²"})
        st.markdown("**All trained regression models**")
        st.dataframe(display_results, use_container_width=True, hide_index=True)
        st.markdown("**RMSE by model**")
        st.bar_chart(results.set_index("Model")[["RMSE"]], color="#27627a", height=300)
        if os.path.isfile("outputs/actual_vs_predicted.png"):
            st.image("outputs/actual_vs_predicted.png", caption="Actual and predicted grades on the holdout split", width=540)
    else:
        st.info("Model comparison results will appear after the training pipeline completes.")
    if os.path.isfile("outputs/eda_overview.png"):
        with st.expander("View training data charts"):
            st.image("outputs/eda_overview.png", caption="Exploratory data overview", use_container_width=True)
            if os.path.isfile("outputs/correlation_heatmap.png"):
                st.image("outputs/correlation_heatmap.png", caption="Feature correlation heatmap", use_container_width=True)

st.markdown("<div style='text-align:center;color:#6b7a89;font-size:.8rem;padding:28px 0 0'>Student Performance Analysis · UCI Student Performance dataset</div>", unsafe_allow_html=True)
