import streamlit as st

# ---------------------------- PAGE CONFIG MUST BE FIRST ----------------------------
st.set_page_config(page_title="Car Price Predictor | AutoValuer Pro", page_icon="🚗", layout="wide")

# ---------------------------- REST OF IMPORTS ----------------------------
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from datetime import datetime
import sqlite3
import io
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# ---------------------------- GENERATE DATA & TRAIN MODEL (on first run) ----------------------------
@st.cache_resource(show_spinner="Training model for first use...")
def get_model_and_data():
    """Generate synthetic car data, train a Linear Regression model, and return model + mappings."""
    
    # 1. Create synthetic dataset (realistic for Indian used cars)
    np.random.seed(42)
    n_samples = 1000
    
    companies = ['Maruti', 'Hyundai', 'Honda', 'Tata', 'Mahindra', 'Toyota', 'Ford', 'Volkswagen']
    models_by_company = {
        'Maruti': ['Swift', 'Dzire', 'Baleno', 'Vitara Brezza', 'Ertiga', 'Alto', 'WagonR', 'Ciaz'],
        'Hyundai': ['i10', 'i20', 'Verna', 'Creta', 'Elantra', 'Santro', 'Grand i10'],
        'Honda': ['City', 'Amaze', 'Jazz', 'WR-V', 'CR-V', 'Civic'],
        'Tata': ['Nexon', 'Harrier', 'Tiago', 'Tigor', 'Altroz', 'Safari'],
        'Mahindra': ['Scorpio', 'XUV500', 'Thar', 'Bolero', 'Marazzo', 'KUV100'],
        'Toyota': ['Innova Crysta', 'Fortuner', 'Camry', 'Yaris', 'Glanza'],
        'Ford': ['EcoSport', 'Figo', 'Aspire', 'Endeavour', 'Freestyle'],
        'Volkswagen': ['Polo', 'Vento', 'Ameo', 'Tiguan', 'Passat']
    }
    # Flatten for name generation
    all_models = []
    for comp, models in models_by_company.items():
        for model in models:
            all_models.append((comp, model))
    
    fuel_types = ['Petrol', 'Diesel', 'CNG']
    
    # Generate random data
    data = []
    for _ in range(n_samples):
        company, model = all_models[np.random.randint(0, len(all_models))]
        year = np.random.randint(2005, 2025)
        kms_driven = np.random.randint(0, 200000)
        fuel = np.random.choice(fuel_types, p=[0.7, 0.25, 0.05])
        
        # Base price logic: newer + less km + diesel = higher price
        base_price = 300000
        age_penalty = (2025 - year) * 15000
        km_penalty = kms_driven * 0.5
        fuel_bonus = 50000 if fuel == 'Diesel' else (0 if fuel == 'Petrol' else -20000)
        brand_factor = len(company) * 5000  # arbitrary
        model_factor = len(model) * 2000
        
        price = max(50000, base_price + brand_factor + model_factor - age_penalty - km_penalty + fuel_bonus)
        price += np.random.randint(-20000, 20000)  # noise
        
        data.append([model, company, year, kms_driven, fuel, int(price)])
    
    df = pd.DataFrame(data, columns=['name', 'company', 'year', 'kms_driven', 'fuel_type', 'price'])
    
    # 2. Preprocess and train model
    X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
    y = df['price']
    
    categorical_features = ['name', 'company', 'fuel_type']
    preprocessor = ColumnTransformer(
        transformers=[('onehotencoder', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )
    
    model = Pipeline(steps=[('columntransformer', preprocessor),
                            ('regressor', LinearRegression())])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Build mappings for UI
    car_to_company = dict(zip(df['name'], df['company']))
    companies_sorted = sorted(df['company'].unique())
    cars_by_company = {comp: sorted(df[df['company'] == comp]['name'].unique()) for comp in companies_sorted}
    
    # Extract fuel categories from onehotencoder
    ohe = model.named_steps['columntransformer'].named_transformers_['onehotencoder']
    fuel_cats = ohe.categories_[2].tolist()  # Index 2 corresponds to fuel_type
    
    return model, car_to_company, companies_sorted, cars_by_company, fuel_cats

# Load everything (cached after first run)
model, car_to_company, COMPANIES, CARS_BY_COMPANY, FUEL_TYPES = get_model_and_data()

# ---------------------------- DATABASE SETUP (SQLite) ----------------------------
DB_NAME = "history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            car_name TEXT,
            company TEXT,
            year INTEGER,
            kms_driven INTEGER,
            fuel_type TEXT,
            predicted_price REAL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            car_a_name TEXT,
            car_a_company TEXT,
            car_a_year INTEGER,
            car_a_kms INTEGER,
            car_a_fuel TEXT,
            car_b_name TEXT,
            car_b_company TEXT,
            car_b_year INTEGER,
            car_b_kms INTEGER,
            car_b_fuel TEXT,
            price_a REAL,
            price_b REAL,
            price_diff REAL,
            diff_percent REAL
        )
    ''')
    conn.commit()
    conn.close()

def insert_prediction(car_name, company, year, kms_driven, fuel_type, price):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO predictions (timestamp, car_name, company, year, kms_driven, fuel_type, predicted_price)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (now, car_name, company, year, kms_driven, fuel_type, price))
    conn.commit()
    conn.close()

def get_all_predictions():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def delete_predictions(ids):
    if not ids:
        return
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.executemany("DELETE FROM predictions WHERE id = ?", [(i,) for i in ids])
    conn.commit()
    conn.close()

def insert_comparison(data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO comparisons (
            timestamp, car_a_name, car_a_company, car_a_year, car_a_kms, car_a_fuel,
            car_b_name, car_b_company, car_b_year, car_b_kms, car_b_fuel,
            price_a, price_b, price_diff, diff_percent
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (now, data['car_a_name'], data['car_a_company'], data['car_a_year'], data['car_a_kms'], data['car_a_fuel'],
          data['car_b_name'], data['car_b_company'], data['car_b_year'], data['car_b_kms'], data['car_b_fuel'],
          data['price_a'], data['price_b'], data['price_diff'], data['diff_percent']))
    conn.commit()
    conn.close()

def get_all_comparisons():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM comparisons ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def delete_comparisons(ids):
    if not ids:
        return
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.executemany("DELETE FROM comparisons WHERE id = ?", [(i,) for i in ids])
    conn.commit()
    conn.close()

init_db()

# ---------------------------- PREDICTION FUNCTION ----------------------------
def predict_price(car_name, company, year, kms_driven, fuel_type):
    if car_to_company.get(car_name) != company:
        raise ValueError(f" '{car_name}' does not belong to {company}.")
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

# ---------------------------- CSS ----------------------------
st.markdown("""
<style>
    /* Dark purple theme */
    .stApp { background: linear-gradient(135deg, #1a2a4f 0%, #0f1a2e 100%); }
    .hero { background: linear-gradient(135deg, #2a3f6e 0%, #1a2a4f 100%); padding: 2rem; border-radius: 32px; color: white; text-align: center; margin-bottom: 2rem; }
    .hero h1 { font-size: 3rem; font-weight: 800; }
    .card { background: rgba(248,250,252,0.95); border-radius: 28px; padding: 1.8rem; color: #0f172a; }
    .pred-card { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); border-radius: 32px; padding: 1.8rem; text-align: center; color: white; }
    .pred-number { font-size: 2.8rem; font-weight: 800; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a2a4f 0%, #0f1a2e 100%); }
    .stButton>button { background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%); color: white; border-radius: 40px; }
    .footer { text-align: center; color: #cbd5e1; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------- HERO ----------------------------
st.markdown("""
<div class="hero">
    <h1>🚗 AutoValuer Pro</h1>
    <p>AI-powered used car price estimator – instant, accurate, reliable.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------- SIDEBAR ----------------------------
with st.sidebar:
    st.markdown("## Model Intelligence")
    st.markdown(f"""
    <div style="background: #2a3f6ecc; border-radius: 20px; padding: 1rem;">
        <b>Algorithm:</b> Linear Regression (trained on synthetic data)<br>
        <b>Features:</b> Model, Company, Year, KM, Fuel<br>
        <b>Samples:</b> 1000 generated<br>
        <b>Last trained:</b> {datetime.now().strftime("%Y-%m-%d")}
    </div>
    """, unsafe_allow_html=True)
    st.info("Single file – no external CSV or pickle needed.\n\nAll history stored locally in history.db.\n\nClick rows to select & delete.")

# ---------------------------- TABS ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Single Prediction", "Compare Cars", "Prediction History", "Comparison History"])

# ---------------------------- TAB 1: Single Prediction ----------------------------
with tab1:
    colA, colB = st.columns([2,1])
    with colA:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            company = st.selectbox("Company", COMPANIES, key="comp_single")
            cars = CARS_BY_COMPANY.get(company, [])
            if not cars:
                st.error(f"No models for {company}")
                car_name = ""
            else:
                car_name = st.selectbox("Car Model", cars, key="car_single")
            year = st.number_input("Year", 1990, datetime.now().year, 2018)
            kms = st.number_input("Kilometers driven", 0, step=500, value=30000)
            fuel = st.selectbox("Fuel Type", FUEL_TYPES)
            if st.button("Predict & Save", type="primary", use_container_width=True):
                if car_name:
                    try:
                        price = predict_price(car_name, company, year, kms, fuel)
                        insert_prediction(car_name, company, year, kms, fuel, price)
                        st.session_state["last_price"] = price
                        st.session_state["last_input"] = {"car": car_name, "company": company, "year": year, "kms": kms, "fuel": fuel}
                        st.success("Saved!")
                    except ValueError as e:
                        st.error(e)
            st.markdown('</div>', unsafe_allow_html=True)
    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Estimated Price")
        if "last_price" in st.session_state:
            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-number">{format_price(st.session_state['last_price'])}</div>
                <div>{st.session_state['last_input']['year']} {st.session_state['last_input']['company']} {st.session_state['last_input']['car']}<br>{st.session_state['last_input']['kms']:,} km | {st.session_state['last_input']['fuel']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Predict a car to see the price.")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------- TAB 2: Compare Cars ----------------------------
with tab2:
    st.markdown("### Compare Two Cars")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        comp_a = st.selectbox("Company A", COMPANIES, key="comp_a")
        cars_a = CARS_BY_COMPANY.get(comp_a, [])
        car_a = st.selectbox("Model A", cars_a, key="car_a") if cars_a else ""
        year_a = st.number_input("Year A", 1990, datetime.now().year, 2016, key="year_a")
        kms_a = st.number_input("KM A", 0, step=500, value=40000, key="kms_a")
        fuel_a = st.selectbox("Fuel A", FUEL_TYPES, key="fuel_a")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        comp_b = st.selectbox("Company B", COMPANIES, key="comp_b", index=1 if len(COMPANIES)>1 else 0)
        cars_b = CARS_BY_COMPANY.get(comp_b, [])
        car_b = st.selectbox("Model B", cars_b, key="car_b") if cars_b else ""
        year_b = st.number_input("Year B", 1990, datetime.now().year, 2019, key="year_b")
        kms_b = st.number_input("KM B", 0, step=500, value=20000, key="kms_b")
        fuel_b = st.selectbox("Fuel B", FUEL_TYPES, key="fuel_b")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Compare & Save", type="primary", use_container_width=True):
        if car_a and car_b:
            try:
                price_a = predict_price(car_a, comp_a, year_a, kms_a, fuel_a)
                price_b = predict_price(car_b, comp_b, year_b, kms_b, fuel_b)
                diff = price_b - price_a
                diff_percent = (diff / price_a) * 100 if price_a != 0 else 0
                comp_data = {
                    'car_a_name': car_a, 'car_a_company': comp_a, 'car_a_year': year_a, 'car_a_kms': kms_a, 'car_a_fuel': fuel_a,
                    'car_b_name': car_b, 'car_b_company': comp_b, 'car_b_year': year_b, 'car_b_kms': kms_b, 'car_b_fuel': fuel_b,
                    'price_a': price_a, 'price_b': price_b, 'price_diff': diff, 'diff_percent': diff_percent
                }
                insert_comparison(comp_data)
                st.success("Comparison saved!")
                colA, colB = st.columns(2)
                colA.metric(car_a, format_price(price_a))
                colB.metric(car_b, format_price(price_b), delta=format_price(diff) if diff>=0 else format_price(diff))
            except Exception as e:
                st.error(str(e))
        else:
            st.error("Select both car models.")

# ---------------------------- HELPER TO GET SELECTED ROWS (CROSS-VERSION COMPATIBLE) ----------------------------
def get_selected_indices(selection):
    """Safely retrieve selected row indices from st.dataframe selection."""
    if selection is None:
        return []
    # Try dictionary-style (most reliable across versions)
    if isinstance(selection, dict):
        return selection.get("rows", [])
    # Try attribute .rows (property)
    if hasattr(selection, "rows"):
        rows_attr = selection.rows
        # If it's a method, call it; otherwise use as is
        if callable(rows_attr):
            try:
                return rows_attr()
            except Exception:
                return []
        elif isinstance(rows_attr, list):
            return rows_attr
    # Fallback
    return []

# ---------------------------- TAB 3: Prediction History ----------------------------
with tab3:
    st.markdown("### Single Prediction History")
    df_pred = get_all_predictions()
    if df_pred.empty:
        st.info("No predictions yet.")
    else:
        display = df_pred.drop(columns=['id'])
        selection = st.dataframe(display,
                                 column_config={"predicted_price": st.column_config.NumberColumn("Price", format="₹ %.2f")},
                                 hide_index=True,
                                 use_container_width=True,
                                 selection_mode="multi-row")
        selected_indices = get_selected_indices(selection)
        selected_ids = df_pred.iloc[list(selected_indices)]['id'].tolist() if selected_indices else []
        col_del, col_csv = st.columns(2)
        with col_del:
            if st.button("Delete Selected", use_container_width=True) and selected_ids:
                delete_predictions(selected_ids)
                st.rerun()
        with col_csv:
            csv = io.StringIO()
            df_pred.drop(columns=['id']).to_csv(csv, index=False)
            st.download_button("Download CSV", csv.getvalue(), file_name="predictions.csv", mime="text/csv", use_container_width=True)
        st.caption(f"Selected: {len(selected_ids)} rows")

# ---------------------------- TAB 4: Comparison History ----------------------------
with tab4:
    st.markdown("### Comparison History")
    df_comp = get_all_comparisons()
    if df_comp.empty:
        st.info("No comparisons yet.")
    else:
        display = df_comp.drop(columns=['id'])
        selection = st.dataframe(display,
                                 column_config={"price_a": "Price A (₹)", "price_b": "Price B (₹)", "price_diff": "Diff (₹)", "diff_percent": "Diff %"},
                                 hide_index=True,
                                 use_container_width=True,
                                 selection_mode="multi-row")
        selected_indices = get_selected_indices(selection)
        selected_ids = df_comp.iloc[list(selected_indices)]['id'].tolist() if selected_indices else []
        col_del, col_csv = st.columns(2)
        with col_del:
            if st.button("Delete Selected Comparisons", use_container_width=True) and selected_ids:
                delete_comparisons(selected_ids)
                st.rerun()
        with col_csv:
            csv = io.StringIO()
            df_comp.drop(columns=['id']).to_csv(csv, index=False)
            st.download_button("Download CSV", csv.getvalue(), file_name="comparisons.csv", mime="text/csv", use_container_width=True)
        st.caption(f"Selected: {len(selected_ids)} rows")

# ---------------------------- FOOTER ----------------------------
st.markdown('<div class="footer">🚗 AutoValuer Pro – Fully self-contained | Data generated on the fly | History stored in SQLite</div>', unsafe_allow_html=True)