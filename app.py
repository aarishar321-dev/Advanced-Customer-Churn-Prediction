import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Advanced Customer Churn Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "customer_churn_logistic_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "customer_churn_preprocessor.pkl")

THRESHOLD = 0.30

# ============================================================
# PREMIUM TELECOM DESIGN
# Deep navy + telecom cyan + electric blue
# ============================================================

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --charcoal: #0F1014;
            --charcoal-2: #15171D;
            --charcoal-3: #1B1D24;
            --burgundy: #0F3D2E;
            --burgundy-2: #1B5E42;
            --coral: #4CAF7D;
            --amber: #8A9A93;
            --accent: #0F3D2E;
            --accent-2: #4CAF7D;
            --text: #F5F2F3;
            --text-soft: #D2CBD0;
            --muted: #9B9299;
            --line: #24312B;
            --surface: #171920;
            --surface-2: #1D2028;
            --danger: #E05262;
            --success: #58B88A;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 88% 5%, rgba(15,61,46,.22), transparent 24%),
                radial-gradient(circle at 5% 85%, rgba(76,175,125,.10), transparent 28%),
                var(--charcoal);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: rgba(15,16,20,.92);
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, #10161370 0%, #0F1310 70%, #0F1014 100%);
            border-right: 1px solid #23342B;
        }

        [data-testid="stSidebar"] * {
            color: #F5F2F3 !important;
        }

        [data-testid="stSidebar"] .stButton button {
            background: #14201A;
            border: 1px solid #2A4A38;
            color: #F5F2F3 !important;
        }

        .block-container {
            max-width: 1380px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        /* HERO */
        .hero {
            position: relative;
            overflow: hidden;
            padding: 34px 38px;
            border-radius: 24px;
            background:
                radial-gradient(circle at 92% 10%, rgba(15,61,46,.30), transparent 25%),
                radial-gradient(circle at 72% 100%, rgba(76,175,125,.20), transparent 28%),
                linear-gradient(135deg, #0E1613 0%, #101A16 58%, #0D1F17 100%);
            box-shadow: 0 20px 55px rgba(0,0,0,.28);
            color: #FFFFFF;
            margin-bottom: 24px;
            border: 1px solid #223A2E;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 280px;
            height: 280px;
            border: 1px solid rgba(76,175,125,.20);
            border-radius: 50%;
            right: -95px;
            top: -100px;
        }

        .eyebrow {
            display: inline-block;
            padding: 6px 11px;
            border-radius: 999px;
            background: rgba(76,175,125,.12);
            border: 1px solid rgba(76,175,125,.32);
            color: #7FD9A6;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: .13em;
            text-transform: uppercase;
            margin-bottom: 13px;
        }

        .hero h1 {
            font-size: 40px;
            line-height: 1.08;
            margin: 0;
            letter-spacing: -1.5px;
            color: #FFFFFF;
        }

        .hero p {
            max-width: 740px;
            color: #C8BEC4;
            font-size: 15px;
            line-height: 1.7;
            margin: 12px 0 0;
        }

        /* SECTION */
        .section-title {
            font-size: 21px;
            font-weight: 800;
            color: #FFFFFF;
            margin: 8px 0 5px;
            letter-spacing: -.4px;
        }

        .section-subtitle {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 16px;
        }

        /* CARDS */
        .metric-card {
            background: linear-gradient(145deg, #191B22, #15171D);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 18px 20px;
            min-height: 112px;
            box-shadow: 0 8px 28px rgba(0,0,0,.20);
        }

        .metric-label {
            color: #A59CA2;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .07em;
        }

        .metric-value {
            color: #FFFFFF;
            font-size: 28px;
            font-weight: 800;
            margin-top: 8px;
        }

        .metric-note {
            color: #7F777F;
            font-size: 11px;
            margin-top: 3px;
        }

        .result-card {
            border-radius: 22px;
            padding: 26px;
            margin-top: 18px;
            border: 1px solid var(--line);
            background: #191B22;
            box-shadow: 0 12px 35px rgba(0,0,0,.22);
        }

        .risk-high {
            border-left: 6px solid var(--danger);
            background: linear-gradient(135deg, #25171D, #191B22);
        }

        .risk-low {
            border-left: 6px solid var(--success);
            background: linear-gradient(135deg, #12241C, #191B22);
        }

        .risk-title {
            font-size: 25px;
            font-weight: 800;
            color: #FFFFFF;
        }

        .risk-description {
            color: #AAA2A8;
            font-size: 14px;
            line-height: 1.65;
            margin-top: 7px;
        }

        .probability {
            font-size: 44px;
            font-weight: 800;
            color: var(--coral);
            letter-spacing: -1.5px;
        }

        .pill {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 800;
            background: rgba(76,175,125,.14);
            color: #7FD9A6;
            border: 1px solid rgba(76,175,125,.28);
        }

        .info-card {
            background: linear-gradient(145deg, #191B22, #15171D);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 20px;
            height: 100%;
        }

        .info-card h4 {
            margin: 0 0 8px;
            color: #FFFFFF;
            font-size: 15px;
        }

        .info-card p {
            margin: 0;
            color: #A49CA2;
            font-size: 13px;
            line-height: 1.65;
        }

        /* STREAMLIT WIDGETS */
        .stButton > button,
        [data-testid="stFormSubmitButton"] button {
            border-radius: 12px;
            border: 0;
            min-height: 48px;
            font-weight: 800;
            background: linear-gradient(90deg, #0F3D2E, #2E7D53);
            color: #FFFFFF !important;
            box-shadow: 0 8px 22px rgba(15,61,46,.35);
        }

        .stButton > button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            border: 0;
            color: #FFFFFF !important;
            background: linear-gradient(90deg, #144A38, #37955F);
        }

        /* Explicit widget-label styling fixes the previously invisible labels. */
        label,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span {
            color: #D9D2D7 !important;
            opacity: 1 !important;
            font-size: 12px !important;
            font-weight: 650 !important;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-radius: 10px !important;
            border: 1px solid #3A3B45 !important;
            background: #111318 !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input,
        .stNumberInput input,
        .stSelectbox input,
        .stTextInput input {
            color: #F5F2F3 !important;
            -webkit-text-fill-color: #F5F2F3 !important;
            font-size: 13px !important;
        }

        div[data-baseweb="select"] svg {
            fill: #B8B0B6 !important;
        }

        [data-baseweb="popover"] {
            background: #1B1D24 !important;
            border: 1px solid #3A3B45 !important;
        }

        [role="option"] {
            color: #F5F2F3 !important;
            background: #1B1D24 !important;
        }

        [role="option"]:hover {
            background: #30202A !important;
        }

        [data-testid="stMetric"] {
            background: #191B22;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 13px 16px;
        }

        [data-testid="stMetricLabel"] {
            color: #A59CA2 !important;
        }

        [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            font-weight: 700;
            color: #938A91 !important;
        }

        .stTabs [aria-selected="true"] {
            color: #4CAF7D !important;
            border-bottom-color: #0F3D2E !important;
        }

        hr {
            border-color: var(--line);
        }

        .footer {
            margin-top: 45px;
            padding-top: 20px;
            border-top: 1px solid var(--line);
            color: #706970;
            font-size: 11px;
            text-align: center;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA / MODEL
# ============================================================

MODEL_COMPARISON = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"],
    "Accuracy": [0.804116, 0.730305, 0.790632, 0.799858],
    "Precision": [0.664430, 0.491525, 0.637631, 0.650327],
    "Recall": [0.529412, 0.465241, 0.489305, 0.532086],
    "F1 Score": [0.589286, 0.478022, 0.553707, 0.585294],
    "ROC-AUC": [0.845984, 0.645147, 0.822676, 0.843105],
})

FINAL_METRICS = {
    "Precision": 0.532710,
    "Recall": 0.762032,
    "F1 Score": 0.627063,
    "ROC-AUC": 0.845984,
}

EXPECTED_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "ServiceCount", "IsNewCustomer",
    "TenureGroup", "MonthlyChargeGroup"
]

FALLBACK_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ],
    "TenureGroup": ["Q1 - Lowest", "Q2", "Q3", "Q4 - Highest"],
    "MonthlyChargeGroup": ["Q1 - Lowest", "Q2", "Q3", "Q4 - Highest"],
}


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        return None, None

    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)


def get_categories(preprocessor):
    categories = {}

    try:
        for _, transformer, columns in preprocessor.transformers_:
            if hasattr(transformer, "named_steps"):
                encoder = transformer.named_steps.get("onehot")

                if encoder is not None and hasattr(encoder, "categories_"):
                    for column, values in zip(columns, encoder.categories_):
                        categories[column] = [str(v) for v in values]
    except Exception:
        pass

    return categories


def options_for(field, categories):
    return categories.get(field, FALLBACK_OPTIONS[field])


def calculate_service_count(internet, security, backup, device, tech, tv, movies):
    if internet == "No":
        return 0

    return sum(
        value == "Yes"
        for value in [security, backup, device, tech, tv, movies]
    )


def calculate_tenure_group(tenure):
    if tenure <= 18:
        return "Q1 - Lowest"
    elif tenure <= 36:
        return "Q2"
    elif tenure <= 54:
        return "Q3"
    return "Q4 - Highest"


def calculate_monthly_group(monthly):
    if monthly <= 35.50:
        return "Q1 - Lowest"
    elif monthly <= 70.35:
        return "Q2"
    elif monthly <= 89.85:
        return "Q3"
    return "Q4 - Highest"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div style="padding:8px 4px 25px;">'
        '<div style="font-size:20px;font-weight:800;margin-top:7px;">'
        'Advanced Customer Churn Prediction'
        '</div>'
        '<div style="font-size:11px;color:#4CAF7D;margin-top:4px;letter-spacing:.06em;">'
        'TELECOM CUSTOMER ANALYTICS'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Navigation")
    st.caption("Use the tabs above to explore the prediction, model performance and business insights.")

    st.markdown("---")
    st.markdown("### Model Snapshot")

    st.markdown(
        """
        **Model**  
        Logistic Regression

        **Operating threshold**  
        0.30

        **ROC-AUC**  
        84.60%

        **Recall**  
        76.20%
        """
    )

    st.markdown("---")
    st.caption("Built as an end-to-end machine learning portfolio project.")


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Telecom • Predictive Analytics</div>
        <h1>Advanced Customer Churn Prediction</h1>
        <p>
            Identify customers who are most likely to leave, understand the drivers
            behind churn, and prioritize retention actions using a trained machine
            learning model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TOP METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Final Model</div>'
        '<div class="metric-value">Logistic</div>'
        '<div class="metric-note">Selected from 4 models</div></div>',
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">ROC-AUC</div>'
        '<div class="metric-value">84.60%</div>'
        '<div class="metric-note">Strong class separation</div></div>',
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Recall</div>'
        '<div class="metric-value">76.20%</div>'
        '<div class="metric-note">At threshold 0.30</div></div>',
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">F1 Score</div>'
        '<div class="metric-value">62.71%</div>'
        '<div class="metric-note">Precision–recall balance</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# MAIN TABS
# ============================================================

tab_predict, tab_performance, tab_insights, tab_project = st.tabs(
    [
        "Churn Prediction",
        "Model Performance",
        "Business Insights",
        "Project",
    ]
)

# ============================================================
# PREDICTION
# ============================================================

with tab_predict:

    st.markdown(
        '<div class="section-title">Customer Risk Assessment</div>'
        '<div class="section-subtitle">Enter customer details to estimate churn probability.</div>',
        unsafe_allow_html=True,
    )

    model, preprocessor = load_artifacts()

    if model is None or preprocessor is None:
        st.error(
            "Model artifacts not found. Add `customer_churn_logistic_model.pkl` "
            "and `customer_churn_preprocessor.pkl` inside the `models` folder."
        )
    else:
        categories = get_categories(preprocessor)

        with st.form("customer_form"):

            st.markdown("#### Customer Profile")

            c1, c2, c3 = st.columns(3)

            with c1:
                gender = st.selectbox("Gender", options_for("gender", categories))
                senior = st.selectbox(
                    "Senior Citizen",
                    [0, 1],
                    format_func=lambda x: "Yes" if x else "No",
                )
                partner = st.selectbox("Partner", options_for("Partner", categories))
                dependents = st.selectbox("Dependents", options_for("Dependents", categories))
                tenure = st.number_input(
                    "Tenure (months)",
                    min_value=0,
                    max_value=100,
                    value=12,
                    step=1,
                )

            with c2:
                phone = st.selectbox("Phone Service", options_for("PhoneService", categories))
                multiple = st.selectbox("Multiple Lines", options_for("MultipleLines", categories))
                internet = st.selectbox("Internet Service", options_for("InternetService", categories))
                security = st.selectbox("Online Security", options_for("OnlineSecurity", categories))
                backup = st.selectbox("Online Backup", options_for("OnlineBackup", categories))
                device = st.selectbox("Device Protection", options_for("DeviceProtection", categories))

            with c3:
                tech = st.selectbox("Tech Support", options_for("TechSupport", categories))
                tv = st.selectbox("Streaming TV", options_for("StreamingTV", categories))
                movies = st.selectbox("Streaming Movies", options_for("StreamingMovies", categories))
                contract = st.selectbox("Contract", options_for("Contract", categories))
                paperless = st.selectbox("Paperless Billing", options_for("PaperlessBilling", categories))
                payment = st.selectbox("Payment Method", options_for("PaymentMethod", categories))

            st.markdown("#### Billing")

            b1, b2 = st.columns(2)

            with b1:
                monthly = st.number_input(
                    "Monthly Charges",
                    min_value=0.0,
                    max_value=200.0,
                    value=70.0,
                    step=0.01,
                )

            with b2:
                total = st.number_input(
                    "Total Charges",
                    min_value=0.0,
                    max_value=10000.0,
                    value=1000.0,
                    step=0.01,
                )

            service_count = calculate_service_count(
                internet, security, backup, device, tech, tv, movies
            )

            is_new_customer = int(tenure <= 6)
            tenure_group = calculate_tenure_group(tenure)
            monthly_group = calculate_monthly_group(monthly)

            st.markdown("#### Engineered Features")

            e1, e2, e3 = st.columns(3)

            with e1:
                st.metric("Additional Services", service_count)

            with e2:
                st.metric("New Customer", "Yes" if is_new_customer else "No")

            with e3:
                st.metric("Risk Threshold", "30%")

            submitted = st.form_submit_button(
                "Predict Customer Churn Risk",
                use_container_width=True,
            )

        if submitted:

            row = pd.DataFrame([{
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": security,
                "OnlineBackup": backup,
                "DeviceProtection": device,
                "TechSupport": tech,
                "StreamingTV": tv,
                "StreamingMovies": movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
                "ServiceCount": service_count,
                "IsNewCustomer": is_new_customer,
                "TenureGroup": tenure_group,
                "MonthlyChargeGroup": monthly_group,
            }])

            row = row[EXPECTED_FEATURES]

            try:
                processed = preprocessor.transform(row)
                probability = float(model.predict_proba(processed)[0, 1])
                prediction = int(probability >= THRESHOLD)

                st.markdown("---")

                if prediction:
                    st.markdown(
                        f"""
                        <div class="result-card risk-high">
                            <div class="pill">HIGH PRIORITY</div>
                            <div class="risk-title">Customer is at high churn risk</div>
                            <div class="risk-description">
                                The predicted churn probability is above the selected
                                business threshold. This customer should be considered
                                for proactive retention outreach.
                            </div>
                            <br>
                            <div class="probability">{probability:.1%}</div>
                            <div style="color:#607086;font-size:12px;">
                                Estimated probability of churn
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card risk-low">
                            <div class="pill">LOW PRIORITY</div>
                            <div class="risk-title">Customer is currently below churn threshold</div>
                            <div class="risk-description">
                                The predicted churn probability is below the selected
                                operating threshold. Continue normal customer engagement
                                and monitor future changes.
                            </div>
                            <br>
                            <div class="probability">{probability:.1%}</div>
                            <div style="color:#607086;font-size:12px;">
                                Estimated probability of churn
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                r1, r2, r3 = st.columns(3)

                with r1:
                    st.metric("Probability", f"{probability:.1%}")

                with r2:
                    st.metric("Threshold", f"{THRESHOLD:.0%}")

                with r3:
                    st.metric("Decision", "CHURN RISK" if prediction else "LOW RISK")

                st.caption(
                    "Threshold 0.30 was selected to improve churn detection. "
                    "On the held-out test set, it achieved approximately 76.2% recall "
                    "and 62.7% F1-score."
                )

            except Exception as exc:
                st.error(
                    "Prediction failed. The saved preprocessing pipeline may not match "
                    "the feature structure used by the Streamlit form."
                )
                st.exception(exc)


# ============================================================
# PERFORMANCE
# ============================================================

with tab_performance:

    st.markdown(
        '<div class="section-title">Model Performance</div>'
        '<div class="section-subtitle">Four classification models were evaluated before selecting the final model.</div>',
        unsafe_allow_html=True,
    )

    chart_df = MODEL_COMPARISON.set_index("Model")[[
        "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"
    ]]

    st.bar_chart(chart_df, height=420)

    st.markdown("#### Comparison")

    display_df = MODEL_COMPARISON.copy()

    for column in display_df.columns[1:]:
        display_df[column] = display_df[column].map(lambda x: f"{x:.2%}")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Why Logistic Regression?")

    st.markdown(
        """
        <div class="info-card">
            <h4>Best overall balance</h4>
            <p>
                Logistic Regression achieved the strongest overall performance across
                the major metrics. XGBoost produced a slightly higher recall, but the
                difference was only 0.27 percentage points. Logistic Regression had
                higher accuracy, precision, F1-score and ROC-AUC, while also remaining
                simpler and easier to interpret.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)

    for col, label, value in [
        (p1, "Precision", FINAL_METRICS["Precision"]),
        (p2, "Recall", FINAL_METRICS["Recall"]),
        (p3, "F1 Score", FINAL_METRICS["F1 Score"]),
        (p4, "ROC-AUC", FINAL_METRICS["ROC-AUC"]),
    ]:
        with col:
            st.metric(label, f"{value:.2%}")


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

with tab_insights:

    st.markdown(
        '<div class="section-title">What Drives Churn?</div>'
        '<div class="section-subtitle">Key patterns identified during exploratory analysis.</div>',
        unsafe_allow_html=True,
    )

    insight_columns = st.columns(3)

    insights = [
        ("Contract", "Month-to-month customers showed substantially higher churn than one-year and two-year customers."),
        ("Payment Method", "Electronic check customers had the highest overall churn rate."),
        ("Internet Service", "Fiber optic customers showed notably higher churn than other internet-service groups."),
        ("Support Services", "Customers without online security or technical support showed higher churn."),
        ("Monthly Charges", "Churned customers had a higher average monthly charge than retained customers."),
        ("Service Count", "Customers with only one additional service had the highest churn rate."),
        ("Paperless Billing", "Paperless-billing customers showed higher churn, particularly among month-to-month customers."),
        ("Tenure", "Longer-tenure customers generally showed stronger retention patterns."),
        ("Gender", "Gender showed almost no meaningful difference in churn rate."),
    ]

    for i, (title, description) in enumerate(insights):
        with insight_columns[i % 3]:
            st.markdown(
                f"""
                <div class="info-card" style="margin-bottom:16px;">
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### Retention Strategy")

    st.markdown(
        """
        <div class="result-card">
            <div class="risk-title" style="font-size:20px;">Turn predictions into action</div>
            <div class="risk-description">
                The model can help a telecom business prioritize customers for
                retention campaigns. High-risk customers can receive proactive
                support, contract incentives, personalized offers or customer-success
                outreach before they leave.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PROJECT
# ============================================================

with tab_project:

    st.markdown(
        '<div class="section-title">About the Project</div>'
        '<div class="section-subtitle">An end-to-end customer churn machine learning portfolio project.</div>',
        unsafe_allow_html=True,
    )

    a1, a2 = st.columns(2)

    with a1:
        st.markdown(
            """
            <div class="info-card">
                <h4>Machine Learning Workflow</h4>
                <p>
                    Data cleaning → EDA → conditional analysis → feature engineering →
                    train/test split → preprocessing pipeline → model training →
                    model comparison → hyperparameter tuning → threshold optimization →
                    final evaluation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown(
            """
            <div class="info-card">
                <h4>Final Model</h4>
                <p>
                    Logistic Regression was selected from Logistic Regression,
                    Decision Tree, Random Forest and XGBoost. A probability threshold
                    of 0.30 was chosen to prioritize recall and improve detection of
                    customers at risk of churn.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-card">
            <h4>Important Interpretation</h4>
            <p>
                The model is a decision-support tool. A churn-risk prediction is not
                a guarantee that a customer will leave. The probability threshold
                should be adjusted if the business changes the relative cost of
                missing a churner versus contacting a customer unnecessarily.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Customer Churn Prediction • Machine Learning Portfolio Project<br>
        Built with Python, Pandas, Scikit-learn, XGBoost & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)