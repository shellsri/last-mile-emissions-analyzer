import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Last-Mile Delivery Emissions Analyzer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .green-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .red-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .orange-card {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a2e;
        border-left: 4px solid #667eea;
        padding-left: 0.7rem;
        margin: 1.5rem 0 1rem 0;
    }
    .rec-box {
        background: #f0fff4;
        border: 1px solid #9ae6b4;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .warn-box {
        background: #fff5f5;
        border: 1px solid #feb2b2;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA GENERATION + MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 1000
    cities         = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Kolkata']
    vehicle_types  = ['Electric Van', 'Diesel Truck', 'Petrol Bike', 'CNG Van', 'Cargo Cycle']
    traffic_levels = ['Low', 'Medium', 'High']
    time_slots     = ['Morning', 'Afternoon', 'Evening', 'Night']

    emission_factors   = {'Electric Van': 0.05, 'Diesel Truck': 0.27,
                          'Petrol Bike': 0.11,  'CNG Van': 0.15, 'Cargo Cycle': 0.00}
    traffic_multiplier = {'Low': 1.0, 'Medium': 1.3, 'High': 1.7}

    city      = np.random.choice(cities, n)
    vehicle   = np.random.choice(vehicle_types, n, p=[0.15, 0.30, 0.25, 0.20, 0.10])
    traffic   = np.random.choice(traffic_levels, n, p=[0.25, 0.45, 0.30])
    time_slot = np.random.choice(time_slots, n)
    distance  = np.round(np.random.uniform(1, 40, n), 2)
    load      = np.round(np.random.uniform(5, 500, n), 1)
    stops     = np.random.randint(1, 15, n)
    failed    = np.random.randint(0, 4, n)

    co2 = np.array([
        round(emission_factors[v] * d * traffic_multiplier[t]
              + (l / 1000) * d * 0.05 + f * 0.8
              + np.random.normal(0, 0.3), 3)
        for v, d, t, l, f in zip(vehicle, distance, traffic, load, failed)
    ])
    co2 = np.clip(co2, 0, None)

    df = pd.DataFrame({
        'City': city, 'Vehicle_Type': vehicle, 'Traffic_Level': traffic,
        'Time_Slot': time_slot, 'Distance_km': distance, 'Load_kg': load,
        'Num_Stops': stops, 'Failed_Deliveries': failed, 'CO2_Emissions_kg': co2
    })
    return df

@st.cache_resource
def train_model(df):
    le = LabelEncoder()
    df_enc = df.copy()
    for col in ['City','Vehicle_Type','Traffic_Level','Time_Slot']:
        df_enc[col] = le.fit_transform(df[col])

    features = ['Distance_km','Load_kg','Num_Stops','Failed_Deliveries',
                'City','Vehicle_Type','Traffic_Level','Time_Slot']
    X = df_enc[features]
    y = df['CO2_Emissions_kg']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    return rf, r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), features

def encode_input(city, vehicle, traffic, time_slot):
    cities    = sorted(['Mumbai','Delhi','Bangalore','Chennai','Hyderabad','Kolkata'])
    vehicles  = sorted(['Electric Van','Diesel Truck','Petrol Bike','CNG Van','Cargo Cycle'])
    traffics  = ['High','Low','Medium']
    times     = ['Afternoon','Evening','Morning','Night']
    return cities.index(city), vehicles.index(vehicle), traffics.index(traffic), times.index(time_slot)

def get_recommendations(vehicle, traffic, failed, distance):
    recs  = []
    warns = []
    if vehicle == 'Diesel Truck':
        warns.append("⚠️ Diesel Truck detected — highest emission vehicle in fleet")
        recs.append("✅ Switch to CNG Van to reduce emissions by ~44%")
        recs.append("✅ Switch to Electric Van to reduce emissions by ~81%")
    if traffic == 'High':
        warns.append("⚠️ High traffic increases emissions by up to 70%")
        recs.append("✅ Reschedule to Morning or Night slot to avoid peak hours")
    if failed > 1:
        warns.append(f"⚠️ {failed} failed deliveries add ~{failed * 0.8:.1f} kg extra CO₂")
        recs.append("✅ Use SMS/call confirmations before delivery to reduce failures")
    if distance < 5 and vehicle not in ['Cargo Cycle', 'Electric Van']:
        warns.append("⚠️ Short route — current vehicle is over-engineered for this distance")
        recs.append("✅ Deploy a Cargo Cycle for routes under 5 km (zero emissions)")
    if not warns:
        recs.append("🌱 This delivery is already well-optimized!")
    return warns, recs

# ─────────────────────────────────────────────
# LOAD DATA + MODEL
# ─────────────────────────────────────────────
df = generate_data()
rf_model, r2, mae, features = train_model(df)

emission_factors = {'Electric Van': 0.05, 'Diesel Truck': 0.27,
                    'Petrol Bike': 0.11,  'CNG Van': 0.15, 'Cargo Cycle': 0.00}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/delivery-truck.png", width=70)
    st.markdown("## 🚚 Emissions Analyzer")
    st.markdown("*IITM FedEx SMART Center — Sustainability Research*")
    st.divider()
    st.markdown(f"**Dataset:** {len(df):,} delivery records")
    st.markdown(f"**Cities covered:** 6 Indian metros")
    st.markdown(f"**ML Model R²:** `{r2:.4f}`")
    st.markdown(f"**Model MAE:** `{mae:.4f} kg CO₂`")
    st.divider()
    st.markdown("**Built by:** Shelly Srivastava")
    st.markdown("**NIT Manipur | B.Tech CSE**")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🚚 Last-Mile Delivery Emissions Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Identify high-emission routes · Predict CO₂ · Get sustainability recommendations</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────
total_co2    = df['CO2_Emissions_kg'].sum()
avg_co2      = df['CO2_Emissions_kg'].mean()
high_em_pct  = (df['CO2_Emissions_kg'] > df['CO2_Emissions_kg'].quantile(0.8)).mean() * 100
diesel_share = (df['Vehicle_Type'] == 'Diesel Truck').mean() * 100

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card red-card"><div class="metric-value">{total_co2:,.0f}</div><div class="metric-label">Total CO₂ Emitted (kg)</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card orange-card"><div class="metric-value">{avg_co2:.2f}</div><div class="metric-label">Avg CO₂ per Delivery (kg)</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{high_em_pct:.0f}%</div><div class="metric-label">High-Emission Deliveries</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card green-card"><div class="metric-value">{diesel_share:.0f}%</div><div class="metric-label">Diesel Truck Share</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA & Insights", "🔮 Predict Emissions", "🌱 Sustainability Impact", "📋 Raw Data"])

# ══════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Emissions by Vehicle Type & Traffic Level</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        vehicle_co2 = df.groupby('Vehicle_Type')['CO2_Emissions_kg'].mean().sort_values(ascending=False)
        colors = ['#e74c3c' if i == 0 else '#667eea' for i in range(len(vehicle_co2))]
        bars = ax.bar(vehicle_co2.index, vehicle_co2.values, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_title('Avg CO₂ by Vehicle Type', fontweight='bold', fontsize=11)
        ax.set_ylabel('Avg CO₂ (kg)')
        ax.tick_params(axis='x', rotation=25)
        for bar, val in zip(bars, vehicle_co2.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f'{val:.2f}', ha='center', fontsize=8, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        traffic_co2 = df.groupby('Traffic_Level')['CO2_Emissions_kg'].mean().reindex(['Low','Medium','High'])
        colors = ['#2ecc71','#f39c12','#e74c3c']
        bars = ax.bar(traffic_co2.index, traffic_co2.values, color=colors, edgecolor='white')
        ax.set_title('Avg CO₂ by Traffic Level', fontweight='bold', fontsize=11)
        ax.set_ylabel('Avg CO₂ (kg)')
        for bar, val in zip(bars, traffic_co2.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.info("💡 **Insight:** Diesel Trucks emit ~5× more CO₂ than Electric Vans. High traffic increases emissions by ~70% vs Low traffic.")

    st.markdown('<div class="section-title">City-wise & Distance Analysis</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        city_total = df.groupby('City')['CO2_Emissions_kg'].sum().sort_values(ascending=False)
        ax.bar(city_total.index, city_total.values, color='#9b59b6', edgecolor='white')
        ax.set_title('Total CO₂ by City', fontweight='bold', fontsize=11)
        ax.set_ylabel('Total CO₂ (kg)')
        ax.tick_params(axis='x', rotation=25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(6, 4))
        palette = {'Diesel Truck':'#e74c3c','Petrol Bike':'#e67e22',
                   'CNG Van':'#f1c40f','Electric Van':'#2ecc71','Cargo Cycle':'#3498db'}
        for vtype, grp in df.groupby('Vehicle_Type'):
            ax.scatter(grp['Distance_km'], grp['CO2_Emissions_kg'],
                       alpha=0.35, s=15, label=vtype, color=palette[vtype])
        ax.set_title('Distance vs CO₂ by Vehicle', fontweight='bold', fontsize=11)
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('CO₂ (kg)')
        ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-title">Feature Importance — What Drives Emissions Most?</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values()
    colors = ['#e74c3c' if i == importances.index[-1] else '#667eea' for i in importances.index]
    importances.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.set_title('Random Forest Feature Importance', fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.info("💡 **Insight:** Vehicle_Type and Distance_km are the two biggest drivers of CO₂ — optimizing these gives the maximum sustainability impact.")

# ══════════════════════════════════════════════
# TAB 2 — PREDICT
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Configure a Delivery & Predict its CO₂ Footprint</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        city      = st.selectbox("🏙️ City",         ['Mumbai','Delhi','Bangalore','Chennai','Hyderabad','Kolkata'])
        vehicle   = st.selectbox("🚛 Vehicle Type",  ['Electric Van','Diesel Truck','Petrol Bike','CNG Van','Cargo Cycle'])
        traffic   = st.selectbox("🚦 Traffic Level", ['Low','Medium','High'])
        time_slot = st.selectbox("🕐 Time Slot",     ['Morning','Afternoon','Evening','Night'])

    with col_b:
        distance = st.slider("📍 Distance (km)",       1.0, 40.0, 15.0, 0.5)
        load     = st.slider("📦 Load Weight (kg)",    5.0, 500.0, 150.0, 5.0)
        stops    = st.slider("🛑 Number of Stops",     1, 15, 5)
        failed   = st.slider("❌ Failed Deliveries",   0, 3, 0)

    if st.button("🔮 Predict CO₂ Emissions", use_container_width=True, type="primary"):
        city_enc, veh_enc, trf_enc, time_enc = encode_input(city, vehicle, traffic, time_slot)

        row = pd.DataFrame([{
            'Distance_km': distance, 'Load_kg': load, 'Num_Stops': stops,
            'Failed_Deliveries': failed, 'City': city_enc,
            'Vehicle_Type': veh_enc, 'Traffic_Level': trf_enc, 'Time_Slot': time_enc
        }])
        predicted_co2 = rf_model.predict(row)[0]
        predicted_co2 = max(0, predicted_co2)

        # Benchmark comparison
        best_row = pd.DataFrame([{
            'Distance_km': distance, 'Load_kg': load, 'Num_Stops': stops,
            'Failed_Deliveries': 0,
            'City': city_enc,
            'Vehicle_Type': sorted(['Electric Van','Diesel Truck','Petrol Bike','CNG Van','Cargo Cycle']).index('Cargo Cycle') if distance < 5 else sorted(['Electric Van','Diesel Truck','Petrol Bike','CNG Van','Cargo Cycle']).index('Electric Van'),
            'Traffic_Level': ['High','Low','Medium'].index('Low'),
            'Time_Slot': ['Afternoon','Evening','Morning','Night'].index('Morning')
        }])
        best_co2 = max(0, rf_model.predict(best_row)[0])
        saving   = max(0, predicted_co2 - best_co2)

        st.divider()
        r1, r2_col, r3 = st.columns(3)
        with r1:
            color = "🔴" if predicted_co2 > df['CO2_Emissions_kg'].quantile(0.8) else ("🟡" if predicted_co2 > df['CO2_Emissions_kg'].mean() else "🟢")
            st.metric(f"{color} Predicted CO₂", f"{predicted_co2:.3f} kg")
        with r2_col:
            st.metric("🌿 Optimized Scenario", f"{best_co2:.3f} kg")
        with r3:
            st.metric("💰 Potential Saving", f"{saving:.3f} kg", delta=f"-{saving:.3f} kg" if saving > 0 else "Already optimal")

        # Recommendations
        warns, recs = get_recommendations(vehicle, traffic, failed, distance)
        if warns:
            st.markdown('<div class="section-title">⚠️ Issues Detected</div>', unsafe_allow_html=True)
            for w in warns:
                st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">✅ Recommendations</div>', unsafe_allow_html=True)
        for r in recs:
            st.markdown(f'<div class="rec-box">{r}</div>', unsafe_allow_html=True)

        # CO2 equivalent
        trees = predicted_co2 / (21/365)
        st.info(f"🌳 This delivery's CO₂ requires **{trees:.1f} tree-days** to offset. At 1,000 deliveries/day, switching to optimal saves **{saving*365000:.0f} kg CO₂/year**.")

# ══════════════════════════════════════════════
# TAB 3 — SUSTAINABILITY IMPACT
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🌱 Fleet Electrification Impact Analysis</div>', unsafe_allow_html=True)

    diesel_df   = df[df['Vehicle_Type'] == 'Diesel Truck']
    current_em  = diesel_df['CO2_Emissions_kg'].sum()
    cng_saving  = current_em * (1 - 0.15/0.27)
    elec_saving = current_em * (1 - 0.05/0.27)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🚛 Current (Diesel) Total CO₂", f"{current_em:,.1f} kg")
    with c2:
        st.metric("🔄 Savings if → CNG Van",      f"{cng_saving:,.1f} kg", delta=f"-{cng_saving/current_em*100:.0f}%")
    with c3:
        st.metric("⚡ Savings if → Electric Van", f"{elec_saving:,.1f} kg", delta=f"-{elec_saving/current_em*100:.0f}%")

    # Failed deliveries impact
    st.markdown('<div class="section-title">❌ Failed Delivery CO₂ Cost</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        failed_co2 = df.groupby('Failed_Deliveries')['CO2_Emissions_kg'].mean()
        ax.bar(failed_co2.index, failed_co2.values,
               color=['#2ecc71','#f39c12','#e67e22','#e74c3c'], edgecolor='white')
        ax.set_title('Avg CO₂ vs Failed Delivery Attempts', fontweight='bold', fontsize=10)
        ax.set_xlabel('Failed Deliveries')
        ax.set_ylabel('Avg CO₂ (kg)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        time_co2 = df.groupby('Time_Slot')['CO2_Emissions_kg'].mean().reindex(['Morning','Afternoon','Evening','Night'])
        ax.bar(time_co2.index, time_co2.values,
               color=['#3498db','#e67e22','#9b59b6','#2c3e50'], edgecolor='white')
        ax.set_title('Avg CO₂ by Delivery Time Slot', fontweight='bold', fontsize=10)
        ax.set_xlabel('Time Slot')
        ax.set_ylabel('Avg CO₂ (kg)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    total_failed_co2 = df['Failed_Deliveries'].sum() * 0.8
    st.warning(f"📦 Failed deliveries across {len(df):,} records add **{total_failed_co2:,.0f} kg** of avoidable CO₂. Better scheduling could eliminate this entirely.")

    st.markdown('<div class="section-title">🏆 Top 3 Sustainability Actions</div>', unsafe_allow_html=True)
    actions = [
        ("1️⃣ Fleet Electrification", f"Replace all Diesel Trucks with Electric Vans → save **{elec_saving:,.0f} kg CO₂** ({elec_saving/current_em*100:.0f}% reduction)"),
        ("2️⃣ Time-Slot Optimization", "Shift deliveries to Morning/Night to avoid peak traffic → up to **70% emissions reduction** per route"),
        ("3️⃣ Micro-logistics for Short Hauls", "Deploy Cargo Cycles for all routes < 5 km → **zero emissions** for urban last mile")
    ]
    for title, body in actions:
        st.markdown(f'<div class="rec-box"><strong>{title}</strong><br>{body}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — RAW DATA
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Delivery Dataset</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    city_filter    = col1.multiselect("Filter by City",    df['City'].unique(),         default=list(df['City'].unique()))
    vehicle_filter = col2.multiselect("Filter by Vehicle", df['Vehicle_Type'].unique(), default=list(df['Vehicle_Type'].unique()))
    traffic_filter = col3.multiselect("Filter by Traffic", df['Traffic_Level'].unique(),default=list(df['Traffic_Level'].unique()))

    filtered = df[
        df['City'].isin(city_filter) &
        df['Vehicle_Type'].isin(vehicle_filter) &
        df['Traffic_Level'].isin(traffic_filter)
    ]

    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} records**")
    st.dataframe(filtered.style.background_gradient(subset=['CO2_Emissions_kg'], cmap='RdYlGn_r'), use_container_width=True)
    st.download_button("⬇️ Download Filtered Data as CSV",
                       data=filtered.to_csv(index=False),
                       file_name="logistics_emissions_filtered.csv",
                       mime="text/csv")
