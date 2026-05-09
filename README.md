# 🚚 Last-Mile Delivery Emissions Analyzer

A data-driven web application to analyze, predict, and reduce CO₂ emissions from last-mile logistics operations — built as part of sustainability research aligned with the **IIT Madras FedEx SMART Center** focus areas.

**🔗 Live App:** _[Add your Streamlit Cloud link here after deployment]_

---

## 📌 Problem Statement

Last-mile delivery is the most carbon-intensive segment of the supply chain, contributing significantly to urban CO₂ emissions. This tool helps logistics operators:
- Identify which routes, vehicles, and time slots drive the most emissions
- Predict CO₂ output for any delivery configuration using ML
- Get actionable sustainability recommendations to reduce their carbon footprint

---

## 🎯 Features

| Feature | Description |
|---|---|
| 📊 EDA Dashboard | 6 interactive charts analyzing emissions by vehicle, city, traffic, and time |
| 🔮 CO₂ Predictor | Configure any delivery → get instant ML-based emission prediction |
| 🌱 Sustainability Engine | Flags high-emission routes with specific actionable recommendations |
| 📋 Data Explorer | Filter, explore, and download the full delivery dataset |

---

## 🧠 ML Models Used

| Model | R² Score | MAE |
|---|---|---|
| Linear Regression | ~0.92 | ~0.35 kg |
| **Random Forest** ✅ | **~0.98** | **~0.12 kg** |

Random Forest is used for predictions, capturing non-linear interactions between vehicle type, traffic level, load, and route complexity.

---

## 📊 Key Findings

- **Diesel Trucks** emit ~5× more CO₂ than Electric Vans
- **High traffic** increases emissions by up to **70%** compared to low traffic
- Each **failed delivery attempt** adds ~0.8 kg of avoidable CO₂
- **Cargo Cycles** produce zero emissions for routes under 5 km — severely underutilized

### Top 3 Sustainability Recommendations
1. **Fleet Electrification** → Replace Diesel Trucks with Electric Vans: **81% emission reduction**
2. **Time-Slot Optimization** → Shift deliveries to Morning/Night to avoid traffic
3. **Micro-logistics** → Deploy Cargo Cycles for all urban routes < 5 km

---

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **ML/Data:** Scikit-learn, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit Community Cloud

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/shellsri/last-mile-emissions-analyzer.git
cd last-mile-emissions-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                          # Main Streamlit application
├── logistics_emissions_analyzer.ipynb  # Full EDA + ML notebook
├── requirements.txt                # Dependencies
└── README.md
```

---

## 👩‍💻 Author

**Shelly Srivastava**
B.Tech CSE | National Institute of Technology, Manipur
📧 shelly.sri18@gmail.com | [GitHub](https://github.com/shellsri)

---

## 🔭 Future Scope
- Integrate real GPS route data for actual distance and traffic patterns
- Add weather and seasonal features to improve prediction accuracy
- Extend to multi-modal logistics (air + road + rail)
- Build a route optimization API using FastAPI
