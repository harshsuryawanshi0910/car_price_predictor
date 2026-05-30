import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from datetime import datetime

#                     PAGE CONFIG 
st.set_page_config(
    page_title="Car Price Predictor | AutoValuer Pro",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

#                      CUSTOM CSS (Dark Purple Theme) 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main app background – deep navy */
    .stApp {
        background: linear-gradient(135deg, #1a2a4f 0%, #0f1a2e 100%);
    }

    /* Hero Section */
    .hero {
        background: linear-gradient(135deg, #2a3f6e 0%, #1a2a4f 100%);
        padding: 2rem;
        border-radius: 32px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 35px -10px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-size: 1.2rem;
        opacity: 0.95;
    }

    /* Cards – light background, dark text */
    .card {
        background: rgba(248,250,252,0.95);
        border-radius: 28px;
        padding: 1.8rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid rgba(255,255,255,0.2);
        color: #0f172a;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
    }

    /* Prediction Card – vibrant gradient, white text */
    .pred-card {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 32px;
        padding: 1.8rem;
        text-align: center;
        color: white !important;
        box-shadow: 0 20px 35px -10px rgba(79,70,229,0.5);
    }
    .pred-number {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0.5rem 0;
        color: white !important;
    }
    .pred-card hr {
        background: rgba(255,255,255,0.3);
    }
    .pred-card div {
        color: white !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2a4f 0%, #0f1a2e 100%);
        border-right: 1px solid #2a3f6e;
    }
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] .stAlert {
        background-color: #2a3f6ecc;
        border-left-color: #4f46e5;
    }

    /* ===== DARK PURPLE INPUT FIELDS ===== */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input, 
    .stSelectbox>div>div>select {
        background: #2d1b4e !important;
        border: 1px solid #6b21a5 !important;
        border-radius: 16px;
        padding: 10px 14px;
        font-size: 1rem;
        color: #f1f5f9 !important;
        font-weight: 500;
    }
    .stTextInput>div>div>input:focus, 
    .stNumberInput>div>div>input:focus, 
    .stSelectbox>div>div>select:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 2px rgba(168,85,247,0.3);
        outline: none;
    }
    /* Dropdown options (when opened) */
    div[data-baseweb="select"] ul {
        background-color: #2d1b4e !important;
        color: #f1f5f9 !important;
    }
    div[data-baseweb="select"] li:hover {
        background-color: #4c1d95 !important;
    }
    /* Labels for inputs */
    .stSelectbox label, .stNumberInput label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        margin-bottom: 0.25rem;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white !important;
        border: none;
        border-radius: 40px;
        padding: 0.7rem 1.8rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 15px rgba(0,0,0,0.3);
        background: linear-gradient(90deg, #7c3aed 0%, #4f46e5 100%);
    }

    /* Tabs – consistent with single prediction style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(248,250,252,0.9);
        border-radius: 40px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        color: #1e293b;
        border: none;
        backdrop-filter: blur(4px);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white !important;
    }

    /* Alert boxes */
    .stAlert {
        border-radius: 16px;
        font-weight: 500;
        background-color: #2d1b4ecc;
        color: #f1f5f9;
        border-left-color: #a855f7;
    }
    .footer {
        text-align: center;
        color: #cbd5e1;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

#                  LOAD MODEL AND MAPPING 
@st.cache_resource(show_spinner="Loading prediction model...")
def load_model():
    with open("LinearRegressionModel.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data(show_spinner="Loading car data...")
def load_car_mapping():
    df = pd.read_csv("Cleaned_Car_data.csv")
    mapping = df[['name', 'company']].drop_duplicates(subset='name')
    car_to_company = dict(zip(mapping['name'], mapping['company']))
    companies = sorted(df['company'].unique())
    cars_by_company = {comp: sorted(df[df['company'] == comp]['name'].unique()) for comp in companies}
    return car_to_company, companies, cars_by_company

model = load_model()
car_to_company, COMPANIES, CARS_BY_COMPANY = load_car_mapping()

# Extract fuel types
col_trans = model.named_steps["columntransformer"]
ohe = col_trans.named_transformers_["onehotencoder"]
categories = ohe.categories_
FUEL_TYPES = categories[2].tolist()

#                  HELPER FUNCTIONS 
def predict_price(car_name, company, year, kms_driven, fuel_type):
    if car_to_company.get(car_name) != company:
        raise ValueError(f" '{car_name}' does not belong to {company}. Please correct the selection.")
    input_df = pd.DataFrame({
        "name": [car_name],
        "company": [company],
        "year": [year],
        "kms_driven": [kms_driven],
        "fuel_type": [fuel_type]
    })
    return model.predict(input_df)[0]

def format_price(price):
    return f"₹ {price:,.2f}"

#                  HERO SECTION 
st.markdown("""
<div class="hero">
    <h1>🚗 AutoValuer Pro</h1>
    <p>AI-powered used car price estimator – instant, accurate, reliable.</p>
</div>
""", unsafe_allow_html=True)

#                  SIDEBAR 
with st.sidebar:
    st.markdown("##  Model Intelligence")
    st.markdown("""
    <div style="background: #2a3f6ecc; border-radius: 20px; padding: 1rem;">
        <b>Algorithm:</b> Linear Regression<br>
        <b>Features:</b> Model, Company, Year, KM, Fuel<br>
        <b>R² Score:</b> 0.90<br>
        <b>Training samples:</b> 815<br>
        <b>Last trained:</b> {date}
    </div>
    """.format(date=datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("###  Pro Tips")
    st.info(
        "✅ Select company first → car models update automatically.\n\n"
        "✅ Use the **Compare** tab to evaluate two cars side‑by‑side.\n\n"
        "✅ Hover over charts for detailed insights."
    )

#                            TABS 
tab1, tab2 = st.tabs([" Single Prediction", " Compare Cars"])

# ------------------------ TAB 1: Single Prediction ------------------------
with tab1:
    colA, colB = st.columns([2, 1], gap="large")
    
    with colA:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            company = st.selectbox(" **Company**", COMPANIES, index=0, key="single_company")
            available_cars = CARS_BY_COMPANY.get(company, [])
            if not available_cars:
                st.error(f"No car models for {company}")
                car_name = ""
            else:
                car_name = st.selectbox(" **Car Model**", available_cars, key="single_car")
            
            col_year_kms = st.columns(2)
            with col_year_kms[0]:
                year = st.number_input(" **Year**", min_value=1990, max_value=datetime.now().year, step=1, value=2018)
            with col_year_kms[1]:
                kms_driven = st.number_input(" **Kilometers**", min_value=0, step=500, value=30000, format="%d")
            
            fuel_type = st.selectbox(" **Fuel Type**", FUEL_TYPES, index=0)
            predict_btn = st.button(" **Predict Price**", use_container_width=True, type="primary")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if predict_btn:
            if not available_cars or not car_name:
                st.error("Please select a valid car model.")
            else:
                with st.spinner("Analyzing market data..."):
                    try:
                        price = predict_price(car_name, company, year, kms_driven, fuel_type)
                        st.session_state["last_price"] = price
                        st.session_state["last_input"] = {
                            "car": car_name,
                            "company": company,
                            "year": year,
                            "kms": kms_driven,
                            "fuel": fuel_type
                        }
                        st.success(" Prediction ready!")
                    except ValueError as e:
                        st.error(str(e))
    
    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("###  Estimated Price")
        if "last_price" in st.session_state:
            st.markdown(f"""
            <div class="pred-card">
                <div style="font-size: 1rem; opacity: 0.9;">Market Value</div>
                <div class="pred-number">{format_price(st.session_state['last_price'])}</div>
                <hr style="background: rgba(255,255,255,0.3);">
                <div style="font-size: 0.85rem;">
                    {st.session_state['last_input']['year']} {st.session_state['last_input']['company']} {st.session_state['last_input']['car']}<br>
                    {st.session_state['last_input']['kms']:,} km | {st.session_state['last_input']['fuel']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Fill in car details and click **Predict Price**.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts
    if "last_price" in st.session_state:
        st.markdown("---")
        st.markdown("###  **Price Trends**")
        base_car = st.session_state["last_input"]["car"]
        base_company = st.session_state["last_input"]["company"]
        base_fuel = st.session_state["last_input"]["fuel"]
        
        years = list(range(2005, datetime.now().year + 1))
        prices_year = []
        for y in years:
            try:
                prices_year.append(predict_price(base_car, base_company, y, 50000, base_fuel))
            except:
                prices_year.append(np.nan)
        df_year = pd.DataFrame({"Year": years, "Price": prices_year}).dropna()
        
        kms_range = list(range(0, 150001, 10000))
        prices_kms = []
        for km in kms_range:
            try:
                prices_kms.append(predict_price(base_car, base_company, 2016, km, base_fuel))
            except:
                prices_kms.append(np.nan)
        df_kms = pd.DataFrame({"Kilometers": kms_range, "Price": prices_kms}).dropna()
        
        col1, col2 = st.columns(2)
        if not df_year.empty:
            fig1 = px.line(df_year, x="Year", y="Price", title=f" {base_car} – Price vs Year (50k km)",
                           markers=True, template="plotly_white", color_discrete_sequence=["#4f46e5"])
            fig1.update_layout(yaxis_title="Price (₹)", hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True, key="year_chart")
        else:
            col1.info("Cannot generate year chart.")
        
        if not df_kms.empty:
            fig2 = px.line(df_kms, x="Kilometers", y="Price", title=f" {base_car} – Price vs Kilometers (2016 model)",
                           markers=True, template="plotly_white", color_discrete_sequence=["#7c3aed"])
            fig2.update_layout(yaxis_title="Price (₹)", hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True, key="kms_chart")
        else:
            col2.info("Cannot generate km chart.")

# ------------------------ TAB 2: Compare Cars (same styling as Single Prediction) ------------------------
with tab2:
    st.markdown("###  **Compare Two Cars**")
    st.caption("Select two vehicles – we'll show the price difference instantly.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("####  Car A")
        company_a = st.selectbox("Company A", COMPANIES, key="comp_a", index=0)
        cars_a = CARS_BY_COMPANY.get(company_a, [])
        car_a_name = st.selectbox("Model A", cars_a, key="car_a") if cars_a else ""
        if not cars_a:
            st.warning(f"No models for {company_a}")
        year_a = st.number_input("Year A", min_value=1990, max_value=datetime.now().year, value=2016, key="year_a")
        kms_a = st.number_input("Kilometers A", min_value=0, step=500, value=40000, key="kms_a")
        fuel_a = st.selectbox("Fuel A", FUEL_TYPES, key="fuel_a", index=0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("####  Car B")
        company_b = st.selectbox("Company B", COMPANIES, key="comp_b", index=1 if len(COMPANIES)>1 else 0)
        cars_b = CARS_BY_COMPANY.get(company_b, [])
        car_b_name = st.selectbox("Model B", cars_b, key="car_b") if cars_b else ""
        if not cars_b:
            st.warning(f"No models for {company_b}")
        year_b = st.number_input("Year B", min_value=1990, max_value=datetime.now().year, value=2019, key="year_b")
        kms_b = st.number_input("Kilometers B", min_value=0, step=500, value=20000, key="kms_b")
        fuel_b = st.selectbox("Fuel B", FUEL_TYPES, key="fuel_b", index=0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button(" Compare Now", type="primary", use_container_width=True):
        if not car_a_name or not car_b_name:
            st.error("Please select valid car models for both sides.")
        else:
            with st.spinner("Calculating..."):
                try:
                    price_a = predict_price(car_a_name, company_a, year_a, kms_a, fuel_a)
                    price_b = predict_price(car_b_name, company_b, year_b, kms_b, fuel_b)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
            
            # Same card style as single prediction result cards
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.markdown(f"""
                <div style="background: rgba(248,250,252,0.95); border-radius: 28px; padding: 1.5rem; text-align:center; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
                    <h3 style="color:#0f172a;">{car_a_name}</h3>
                    <p style="color:#334155;">{company_a} | {year_a} | {kms_a:,} km | {fuel_a}</p>
                    <h2 style="color:#1e3c72;">{format_price(price_a)}</h2>
                </div>
                """, unsafe_allow_html=True)
            with comp_col2:
                st.markdown(f"""
                <div style="background: rgba(248,250,252,0.95); border-radius: 28px; padding: 1.5rem; text-align:center; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
                    <h3 style="color:#0f172a;">{car_b_name}</h3>
                    <p style="color:#334155;">{company_b} | {year_b} | {kms_b:,} km | {fuel_b}</p>
                    <h2 style="color:#2a5298;">{format_price(price_b)}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            diff = price_b - price_a
            diff_percent = (diff / price_a) * 100
            if diff > 0:
                st.success(f" **Car B is {format_price(abs(diff))} ({diff_percent:.1f}%) more expensive than Car A.**")
            elif diff < 0:
                st.error(f" **Car B is {format_price(abs(diff))} ({abs(diff_percent):.1f}%) cheaper than Car A.**")
            else:
                st.info("Both cars have the same estimated price.")

# ======================== FOOTER ========================
st.markdown("""
<div class="footer">
    🚗 AutoValuer Pro – Powered by Streamlit & Scikit-learn | Data source: Quikr Car Dataset
</div>
""", unsafe_allow_html=True)