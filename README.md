
# Smart Water Quality Monitoring — Random Forest 

### 🌍 Real-Time Cloud AI Classification System
An advanced machine learning solution designed to evaluate drinking water safety based on **World Health Organization (WHO)** standards. This repository contains the standalone core predictive model—a highly optimized **Random Forest Classifier** equipped with a dynamic **Confidence Threshold Guard** to seamlessly navigate gray-zone anomalies and borderline contamination risks.

<img width="1324" height="732" alt="Screenshot 2026-06-12 at 22 54 40" src="https://github.com/user-attachments/assets/860eddd7-884f-48fc-9fa9-2717b00f05f8" />


---

##  Core AI Features
* **Random Forest Core Classifier** Powered by an ensemble of 100 decision trees configured with optimized depth boundaries to prevent model overfitting.
* **Confidence Guard Mechanism:** Implements a strict $70\%$ certainty threshold (`CONFIDENCE_THRESHOLD = 70.0`). If the voting consensus among the trees drops below $70\%$, the system proactively flags the sample as `🟡 SUSPECT — UNCERTAIN` instead of outputting a weak guess.
* **Border-Zone Augmentation:** Features custom synthetic data injection directly at the fractional edge of WHO boundaries ($+0.01$ margin shifts). This effectively resolves data blind spots and eliminates classification bias.
---

## ML Boundary Targets (WHO baselines)

| Parameter | Healthy Lower Limit | Healthy Upper Limit | Unit |
| :--- | :---: | :---: | :---: |
| **pH** | 6.5 | 8.5 | *Scalar* |
| **Turbidity** | 0.0 | 4.0 | NTU |
| **Water Temperature** | 10.0 | 35.0 | °C |
| **Electrical Conductivity (EC)** | 0.0 | 1500.0 | µS/cm |
| **Dissolved Oxygen (DO)** | 4.0 | 20.0 | mg/L |

---


---

## 📊 Live Boundary Verification Logs

The table below illustrates exactly how the Random Forest engine manages fractional variances and multi-variable edge cases:

```text
 ============================================================
Gray and critical case testing
============================================================

    pH   Turb   Temp      EC    DO               Decision    Trust           Class
──────────────────────────────────────────────────────────────────────────────────
  8.48   1.50  24.00   500.0  6.50        ✅ SAFE — Normal    84.8%      RF + Rules
       ↳ Safe on the edge (pH=8.48)

  8.53   1.20  23.50   480.0  6.80     ⚠️  ALARM — UNSAFE   100.0%      Rule-Based
       ↳ Unsafe (pH=8.53 > 8.5)
         • pH: 8.530  >  WHO Maximum (8.5)

  7.20   3.91  26.00   400.0  6.00  🟡 SUSPECT — UNCERTAIN    63.5%    RF-Uncertain
       ↳ Safe on the edge (Turb=3.91)
         • Model confidence is low — manual review is recommended

  7.40   4.15  25.80   450.0  5.80     ⚠️  ALARM — UNSAFE   100.0%      Rule-Based
       ↳ Unsafe(Turb=4.15 > 4.0)
         • Turbidity: 4.150 NTU >  WHO Maximum (4.0)

  8.45   3.85  34.00  1450.0  5.10  🟡 SUSPECT — UNCERTAIN    56.8%    RF-Uncertain
       ↳ Gray area (all transactions are on the edge)
         • Model confidence is low — manual review is recommended



```

## 🛠️ How to Deploy Locally

### 1. Prerequisites

Ensure you have a modern Python 3.x stack configured on your workstation (fully compatible with macOS/Linux/Windows shell).

### 2. Installation

Clone the workspace repository and install the data-science dependencies:

```bash
git clone [https://github.com/Zahra11Mosbal11/water-quality-monitor.git](https://github.com/Zahra11Mosbal11/water-quality-monitor.git)
cd water-quality-monitor
pip install numpy pandas scikit-learn requests matplotlib seaborn

```

### 3. Execution

Launch the local inference loop:

```bash
python main.py

```

---

## 📂 Project Structure

```text
├── cloud_engine/
│   ├── main.py                 # Random Forest Core Classifier & Training Pipeline
│   └── requirements.txt        # Managed Python environment dependencies
└── README.md                   # Machine Learning Documentation

```

---

## 👩‍💻 Authors & Academic Affiliation

* **Alawya Mozamel Hassan  Albajory**
* **Safaa Taj Aldeen Hassan Ahmed**
* **Shahd Anwar Abdulrahman Mohammed Noor**
* **Zahra Suliman Abdalla Musbel**
  
* Engineered as an advanced graduation project under professional validation at **SUST (Sudan University of Science and Technology)**.


