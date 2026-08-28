
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Analytics Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "analytics",
    "student_performance.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "student_model.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):
        st.error(
            f"Dataset not found!\n\nExpected location:\n{DATA_PATH}"
        )
        st.stop()

    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model not found!\n\nExpected location:\n{MODEL_PATH}"
        )
        st.stop()

    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777777;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        text-align: center;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        text-align: center;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎓 Student Analytics")

st.sidebar.write(
    "Data Science & Performance Analytics"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Analytics Dashboard",
        "🔮 Student Prediction",
        "📋 Data Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "This application analyzes student performance "
    "and uses machine learning to estimate final scores."
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 Student Performance Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Data-driven insights and machine learning based performance prediction'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1: ANALYTICS DASHBOARD
# ============================================================

if page == "📊 Analytics Dashboard":

    st.header("📊 Analytics Dashboard")

    st.write(
        "Explore student performance, important factors, "
        "distributions and relationships in the dataset."
    )

    st.divider()

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    total_students = len(df)
    avg_score = df["FinalScore"].mean()
    avg_attendance = df["Attendance"].mean()
    avg_study = df["StudyHours"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👨‍🎓 Total Students",
        f"{total_students:,}"
    )

    col2.metric(
        "📈 Average Final Score",
        f"{avg_score:.2f}"
    )

    col3.metric(
        "📝 Average Attendance",
        f"{avg_attendance:.2f}%"
    )

    col4.metric(
        "📚 Average Study Hours",
        f"{avg_study:.2f}"
    )

    st.divider()

    # --------------------------------------------------------
    # SCORE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("📈 Final Score Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))

    sns.histplot(
        df["FinalScore"],
        bins=30,
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Final Score")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Final Scores")

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE CATEGORIES
    # --------------------------------------------------------

    st.subheader("🎯 Performance Categories")

    def category(score):

        if score >= 85:
            return "Excellent"

        elif score >= 70:
            return "Good"

        elif score >= 50:
            return "Average"

        else:
            return "Needs Improvement"

    category_counts = (
        df["FinalScore"]
        .apply(category)
        .value_counts()
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Student Distribution")

        st.bar_chart(category_counts)

    with col2:

        st.write("### Category Summary")

        category_table = pd.DataFrame({
            "Performance Level": category_counts.index,
            "Students": category_counts.values
        })

        category_table["Percentage"] = (
            category_table["Students"]
            / total_students
            * 100
        ).round(2)

        st.dataframe(
            category_table,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # --------------------------------------------------------
    # STUDY HOURS VS FINAL SCORE
    # --------------------------------------------------------

    st.subheader("📚 Study Hours vs Final Score")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.scatterplot(
        data=df,
        x="StudyHours",
        y="FinalScore",
        alpha=0.5,
        ax=ax
    )

    ax.set_title(
        "Relationship Between Study Hours and Final Score"
    )

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # --------------------------------------------------------
    # CORRELATION HEATMAP
    # --------------------------------------------------------

    st.subheader("🔥 Feature Correlation Analysis")

    correlation = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title("Correlation Matrix")

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # --------------------------------------------------------
    # IMPORTANT INSIGHTS
    # --------------------------------------------------------

    st.subheader("💡 Key Data Insights")

    correlations = (
        df.corr(numeric_only=True)["FinalScore"]
        .drop("FinalScore")
        .sort_values(ascending=False)
    )

    strongest_feature = correlations.index[0]
    strongest_value = correlations.iloc[0]

    st.write(
        f"🔹 **{strongest_feature}** has the strongest "
        f"positive relationship with Final Score "
        f"({strongest_value:.2f})."
    )

    st.write(
        f"🔹 The average student final score is "
        f"**{avg_score:.2f}**."
    )

    st.write(
        f"🔹 Students study an average of "
        f"**{avg_study:.2f} hours per day**."
    )

    st.write(
        f"🔹 Average attendance is "
        f"**{avg_attendance:.2f}%**."
    )


# ============================================================
# PAGE 2: STUDENT PREDICTION
# ============================================================

elif page == "🔮 Student Prediction":

    st.header("🔮 Student Performance Prediction")

    st.write(
        "Enter student information below to estimate "
        "the expected final score."
    )

    st.divider()

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📚 Academic Information")

        study_hours = st.number_input(
            "Study Hours per Day",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5
        )

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )

        previous_score = st.number_input(
            "Previous Score",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=1.0
        )

    with col2:

        st.subheader("📝 Other Information")

        assignment_score = st.number_input(
            "Assignment Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

        sleep_hours = st.number_input(
            "Sleep Hours per Day",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Student Performance",
        use_container_width=True
    ):

        input_data = pd.DataFrame({

            "StudyHours": [study_hours],

            "Attendance": [attendance],

            "PreviousScore": [previous_score],

            "AssignmentScore": [assignment_score],

            "SleepHours": [sleep_hours]

        })

        prediction = model.predict(input_data)[0]

        prediction = max(
            0,
            min(100, prediction)
        )

        # Performance level

        if prediction >= 85:

            performance = "Excellent"
            message = (
                "The student is expected to perform "
                "at an excellent level."
            )

        elif prediction >= 70:

            performance = "Good"
            message = (
                "The student is expected to perform "
                "at a good level."
            )

        elif prediction >= 50:

            performance = "Average"
            message = (
                "The student may require additional "
                "academic support."
            )

        else:

            performance = "Needs Improvement"
            message = (
                "The student may require significant "
                "academic intervention."
            )

        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Final Score",
                f"{prediction:.2f}/100"
            )

        with col2:

            st.metric(
                "Performance Level",
                performance
            )

        st.success(message)

        # ----------------------------------------------------
        # INPUT SUMMARY
        # ----------------------------------------------------

        st.divider()

        st.subheader("📋 Student Input Summary")

        summary = pd.DataFrame({
            "Factor": [
                "Study Hours",
                "Attendance",
                "Previous Score",
                "Assignment Score",
                "Sleep Hours"
            ],

            "Value": [
                f"{study_hours:.1f} hours",
                f"{attendance:.1f}%",
                f"{previous_score:.1f}",
                f"{assignment_score:.1f}",
                f"{sleep_hours:.1f} hours"
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 3: DATA EXPLORER
# ============================================================

elif page == "📋 Data Explorer":

    st.header("📋 Data Explorer")

    st.write(
        "Explore and understand the student performance dataset."
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.divider()

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    st.subheader("🔍 Filter Students")

    minimum_score = st.slider(
        "Minimum Final Score",
        min_value=0.0,
        max_value=100.0,
        value=0.0
    )

    filtered_df = df[
        df["FinalScore"] >= minimum_score
    ]

    st.write(
        f"Showing **{len(filtered_df):,}** students"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )

    st.divider()

    # --------------------------------------------------------
    # STATISTICAL SUMMARY
    # --------------------------------------------------------

    st.subheader("📊 Statistical Summary")

    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    st.subheader("🔎 Data Quality Check")

    missing = df.isnull().sum()

    missing_table = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values
    })

    st.dataframe(
        missing_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Student Performance Analytics | "
    "Python • Pandas • Matplotlib • Seaborn • "
    "Scikit-learn • Streamlit"
)

