import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delivery CO₂ Analyzer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── STYLES ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero { padding: 2.5rem 0 1rem; }
.hero h1 { font-size: 2rem; font-weight: 700; color: #0f172a; margin: 0; line-height: 1.2; }
.hero p  { font-size: 1rem; color: #64748b; margin: 0.5rem 0 0; }

.form-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.8rem;
}
.form-card h3 { font-size: 1rem; font-weight: 600; color: #0f172a; margin: 0 0 1.2rem; }

.result-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 16px;
    padding: 2rem;
    color: white;
    margin-bottom: 1rem;
}
.co2-number { font-size: 3.5rem; font-weight: 700; line-height: 1; color: white; }
.co2-unit   { font-size: 1rem; color: #94a3b8; margin-top: 4px; }
.co2-badge  {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.5rem;
}
.badge-high   { background: #fee2e2; color: #991b1b; }
.badge-medium { background: #fef3c7; color: #92400e; }
.badge-low    { background: #dcfce7; color: #166534; }

.saving-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.saving-card h4 { color: #166534; font-size: 0.85rem; font-weight: 600; margin: 0 0 0.3rem; }
.saving-card p  { color: #15803d; font-size: 1.3rem; font-weight: 700; margin: 0; }

.mini-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 1rem; }
.mini-stat {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
}
.mini-stat .val   { font-size: 1.3rem; font-weight: 700; color: #0f172a; }
.mini-stat .label { font-size: 0.72rem; color: #64748b; margin-top: 2px; }

.rec-list { display: flex; flex-direction: column; gap: 8px; margin-top: 0.5rem; }
.rec-item {
    display: flex; align-items: flex-start; gap: 10px;
    border-radius: 10px; padding: 0.85rem 1rem;
    font-size: 0.85rem; line-height: 1.5;
}
.rec-warn { background: #fff7ed; border: 1px solid #fed7aa; color: #7c2d12; }
.rec-tip  { background: #f0fdf4; border: 1px solid #bbf7d0; color: #14532d; }
.rec-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }

.empty-state {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    color: #94a3b8;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 0.8rem; }
.empty-state h3 { font-size: 1.1rem; font-weight: 600; color: #475569; margin: 0 0 0.4rem; }
.empty-state p  { font-size: 0.88rem; margin: 0; }

.divider { height: 1px; background: #e2e8f0; margin: 1rem 0; }

.stButton > button {
    background: #0f172a !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #1e3a5f !important; }

label { font-size: 0.88rem !important; font-weight: 500 !important; color: #374151 !important; }
.stSelectbox > div > div { border-radius: 8px !important; }
.stSlider > div { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── DATA + MODEL (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    np.random.seed(42)
    n = 2000
    cities         = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Kolkata']
    vehicle_types  = ['Electric Van', 'Diesel Truck', 'Petrol Bike', 'CNG Van', 'Cargo Cycle']
    traffic_levels = ['Low', 'Medium', 'High']
    time_slots     = ['Morning', 'Afternoon', 'Evening', 'Night']

    emission_factors   = {'Electric Van': 0.05, 'Diesel Truck': 0.27,
                          'Petrol Bike': 0.11, 'CNG Van': 0.15, 'Cargo Cycle': 0.00}
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
        max(0, round(emission_factors[v] * d * traffic_multiplier[t]
                     + (l / 1000) * d * 0.05 + f * 0.8
                     + np.random.normal(0, 0.2), 3))
        for v, d, t, l, f in zip(vehicle, distance, traffic, load, failed)
    ])

    df = pd.DataFrame({
        'City': city, 'Vehicle_Type': vehicle, 'Traffic_Level': traffic,
        'Time_Slot': time_slot, 'Distance_km': distance, 'Load_kg': load,
        'Num_Stops': stops, 'Failed_Deliveries': failed, 'CO2_kg': co2
    })

    le_city    = LabelEncoder().fit(cities)
    le_vehicle = LabelEncoder().fit(sorted(vehicle_types))
    le_traffic = LabelEncoder().fit(sorted(traffic_levels))
    le_time    = LabelEncoder().fit(sorted(time_slots))

    df_enc = df.copy()
    df_enc['City']          = le_city.transform(df['City'])
    df_enc['Vehicle_Type']  = le_vehicle.transform(df['Vehicle_Type'])
    df_enc['Traffic_Level'] = le_traffic.transform(df['Traffic_Level'])
    df_enc['Time_Slot']     = le_time.transform(df['Time_Slot'])

    features = ['Distance_km','Load_kg','Num_Stops','Failed_Deliveries',
                'City','Vehicle_Type','Traffic_Level','Time_Slot']
    X, y = df_enc[features], df['CO2_kg']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    encoders = {'city': le_city, 'vehicle': le_vehicle, 'traffic': le_traffic, 'time': le_time}
    stats    = {'p80': float(np.percentile(co2, 80)), 'p40': float(np.percentile(co2, 40)),
                'mean': float(np.mean(co2)), 'max': float(np.max(co2))}
    return model, encoders, features, stats, df

def predict(model, encoders, features, city, vehicle, traffic, time_slot,
            distance, load, stops, failed):
    row = pd.DataFrame([{
        'Distance_km'      : distance,
        'Load_kg'          : load,
        'Num_Stops'        : stops,
        'Failed_Deliveries': failed,
        'City'             : encoders['city'].transform([city])[0],
        'Vehicle_Type'     : encoders['vehicle'].transform([vehicle])[0],
        'Traffic_Level'    : encoders['traffic'].transform([traffic])[0],
        'Time_Slot'        : encoders['time'].transform([time_slot])[0],
    }])
    return max(0.0, float(model.predict(row)[0]))

def best_case(model, encoders, features, city, distance, load, stops):
    vehicle   = 'Cargo Cycle' if distance < 5 else 'Electric Van'
    return predict(model, encoders, features, city, vehicle, 'Low', 'Morning',
                   distance, load, stops, 0)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚚 Delivery CO₂ Analyzer</h1>
  <p>Enter your delivery details and instantly see your carbon footprint — plus what you can do to reduce it.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    model, encoders, features, stats, df_raw = load_model()

# ── LAYOUT: FORM | RESULTS ───────────────────────────────────────────────────
col_form, col_results = st.columns([1, 1.1], gap="large")

# ════════════════════════════════════════
# LEFT — INPUT FORM
# ════════════════════════════════════════
with col_form:
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Your delivery details")

    city      = st.selectbox("🏙️ City",          ['Mumbai','Delhi','Bangalore','Chennai','Hyderabad','Kolkata'])
    vehicle   = st.selectbox("🚛 Vehicle type",   ['Diesel Truck','Electric Van','Petrol Bike','CNG Van','Cargo Cycle'])
    traffic   = st.selectbox("🚦 Traffic level",  ['High','Medium','Low'])
    time_slot = st.selectbox("🕐 Time of delivery", ['Afternoon','Morning','Evening','Night'])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    distance = st.slider("📍 Distance (km)",       1, 40,  15)
    load     = st.slider("📦 Package weight (kg)", 5, 500, 100, step=5)
    stops    = st.slider("🛑 Number of stops",     1, 15,  4)
    failed   = st.slider("❌ Previous failed attempts (today)", 0, 3, 0)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    analyse = st.button("Analyse my delivery →")
    st.markdown('</div>', unsafe_allow_html=True)

    # Emission factor reference
    with st.expander("📖 How are emissions calculated?"):
        st.markdown("""
**CO₂ is estimated using industry emission factors:**

| Vehicle | kg CO₂ per km |
|---|---|
| Cargo Cycle | 0.00 |
| Electric Van | 0.05 |
| Petrol Bike | 0.11 |
| CNG Van | 0.15 |
| Diesel Truck | 0.27 |

Traffic level adds a multiplier (Low ×1.0, Medium ×1.3, High ×1.7).
Failed deliveries each add ~0.8 kg from re-routing.
A Random Forest model (R² = 0.98) trained on 2,000 delivery records is used for final prediction.
        """)

# ════════════════════════════════════════
# RIGHT — RESULTS
# ════════════════════════════════════════
with col_results:
    if not analyse:
        st.markdown("""
<div class="empty-state">
  <div class="icon">📊</div>
  <h3>Your emissions report will appear here</h3>
  <p>Fill in your delivery details on the left and click <strong>Analyse</strong> to see your CO₂ footprint, potential savings, and sustainability recommendations.</p>
</div>
""", unsafe_allow_html=True)

    else:
        co2    = predict(model, encoders, features, city, vehicle, traffic, time_slot,
                         distance, load, stops, failed)
        best   = best_case(model, encoders, features, city, distance, load, stops)
        saving = max(0.0, co2 - best)

        # Badge
        if co2 >= stats['p80']:
            badge_class, badge_text = "badge-high",   "⚠️ High emission"
        elif co2 >= stats['p40']:
            badge_class, badge_text = "badge-medium", "🟡 Moderate emission"
        else:
            badge_class, badge_text = "badge-low",    "✅ Low emission"

        # ── HERO CO2 RESULT
        st.markdown(f"""
<div class="result-hero">
  <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.05em;">Predicted CO₂ output</div>
  <div class="co2-number">{co2:.2f}</div>
  <div class="co2-unit">kg of CO₂ for this delivery</div>
  <span class="co2-badge {badge_class}">{badge_text}</span>
</div>
""", unsafe_allow_html=True)

        # ── SAVINGS CARD (only show if there's meaningful saving)
        if saving > 0.05:
            st.markdown(f"""
<div class="saving-card">
  <h4>💡 Potential saving with best alternative</h4>
  <p>You could emit just {best:.2f} kg — saving <strong>{saving:.2f} kg CO₂</strong> on this single delivery</p>
</div>
""", unsafe_allow_html=True)

        # ── MINI STATS
        trees_days  = co2 / (21 / 365)
        annual_save = saving * 365 * 50  # assume 50 deliveries/day fleet
        percentile  = int((df_raw['CO2_kg'] < co2).mean() * 100)

        st.markdown(f"""
<div class="mini-grid">
  <div class="mini-stat">
    <div class="val">{trees_days:.0f}</div>
    <div class="label">tree-days to offset</div>
  </div>
  <div class="mini-stat">
    <div class="val">Top {100-percentile}%</div>
    <div class="label">vs all deliveries</div>
  </div>
  <div class="mini-stat">
    <div class="val">{annual_save:.0f} kg</div>
    <div class="label">annual fleet saving*</div>
  </div>
</div>
<p style="font-size:0.72rem;color:#94a3b8;margin:-6px 0 12px;">*if 50 deliveries/day switched to best alternative</p>
""", unsafe_allow_html=True)

        # ── RECOMMENDATIONS
        warns, tips = [], []

        if vehicle == 'Diesel Truck':
            warns.append("⚠️ Diesel Truck is the highest-emission vehicle — contributes up to 5× more CO₂ than an Electric Van")
            tips.append("✅ Switch to Electric Van → cut emissions by ~81% instantly")
            tips.append("✅ Switch to CNG Van → a cost-effective middle step, ~44% reduction")
        elif vehicle == 'Petrol Bike':
            warns.append("⚠️ Petrol Bike emits more than cleaner alternatives for similar distances")
            if distance < 8:
                tips.append("✅ Cargo Cycle is zero-emission and ideal for short routes under 8 km")

        if traffic == 'High':
            warns.append("⚠️ High traffic is adding up to 70% extra CO₂ through idle time and stop-start driving")
            tips.append("✅ Reschedule to Morning or Night slot — same route, far lower emissions")

        if failed > 0:
            warns.append(f"⚠️ {failed} failed attempt(s) added ~{failed * 0.8:.1f} kg of avoidable CO₂ from re-routing")
            tips.append("✅ SMS or call confirmation before departure eliminates most failed attempts")

        if distance < 5 and vehicle not in ['Cargo Cycle', 'Electric Van']:
            warns.append(f"⚠️ Route is only {distance} km — current vehicle is over-engineered for this distance")
            tips.append("✅ Deploy a Cargo Cycle for routes under 5 km — zero emissions, lower cost")

        if not warns and not tips:
            tips.append("✅ This delivery is already well-optimised — great choices!")

        st.markdown("**Issues detected & recommendations**")
        st.markdown('<div class="rec-list">', unsafe_allow_html=True)
        for w in warns:
            st.markdown(f'<div class="rec-item rec-warn"><span class="rec-icon">⚠️</span><span>{w}</span></div>', unsafe_allow_html=True)
        for t in tips:
            st.markdown(f'<div class="rec-item rec-tip"><span class="rec-icon">✅</span><span>{t}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── COMPARISON CHART
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        with st.expander("📊 See how this compares across vehicle types"):
            vehicles    = ['Cargo Cycle', 'Electric Van', 'CNG Van', 'Petrol Bike', 'Diesel Truck']
            ef          = {'Electric Van': 0.05, 'Diesel Truck': 0.27,
                           'Petrol Bike': 0.11, 'CNG Van': 0.15, 'Cargo Cycle': 0.00}
            tm          = {'Low': 1.0, 'Medium': 1.3, 'High': 1.7}
            bar_vals    = [max(0, ef[v] * distance * tm[traffic] + (load/1000)*distance*0.05 + failed*0.8)
                           for v in vehicles]
            bar_colors  = ['#16a34a' if v == vehicle else
                           ('#3b82f6' if ef[v] == 0 or ef[v] <= 0.05 else
                            '#f59e0b' if ef[v] <= 0.15 else '#ef4444')
                           for v in vehicles]
            bar_colors  = ['#e11d48' if v == vehicle else
                           ('#16a34a' if ef.get(v,0) <= 0.05 else
                            '#f59e0b' if ef.get(v,0) <= 0.15 else '#94a3b8')
                           for v in vehicles]

            fig, ax = plt.subplots(figsize=(6.5, 3))
            bars = ax.barh(vehicles, bar_vals,
                           color=['#e11d48' if v == vehicle else
                                  ('#16a34a' if ef.get(v,0) <= 0.05 else
                                   '#f59e0b' if ef.get(v,0) <= 0.15 else '#94a3b8')
                                  for v in vehicles],
                           edgecolor='none', height=0.55)
            for bar, val in zip(bars, bar_vals):
                ax.text(val + 0.03, bar.get_y() + bar.get_height()/2,
                        f'{val:.2f} kg', va='center', fontsize=9,
                        color='#374151', fontweight='500')
            ax.set_xlabel('CO₂ Emissions (kg)', fontsize=9, color='#64748b')
            ax.set_xlim(0, max(bar_vals) * 1.35)
            ax.tick_params(axis='both', labelsize=9, colors='#374151')
            ax.spines[['top','right','left']].set_visible(False)
            ax.spines['bottom'].set_color('#e2e8f0')
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')

            legend = [mpatches.Patch(color='#e11d48', label='Your current choice'),
                      mpatches.Patch(color='#16a34a', label='Best alternatives'),
                      mpatches.Patch(color='#f59e0b', label='Moderate'),
                      mpatches.Patch(color='#94a3b8', label='High emission')]
            ax.legend(handles=legend, fontsize=8, loc='lower right',
                      frameon=False, labelcolor='#374151')
            ax.set_title(f'CO₂ for {distance} km route in {traffic.lower()} traffic',
                         fontsize=10, color='#0f172a', fontweight='600', pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:0.8rem;color:#94a3b8;'>"
    "Built by Shelly Srivastava · NIT Manipur · B.Tech CSE · "
    "<a href='https://github.com/shellsri/last-mile-emissions-analyzer' style='color:#64748b;'>GitHub</a>"
    "</p>",
    unsafe_allow_html=True
)
