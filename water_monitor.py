# Water Quality Monitoring System
#  Random Forest
# Data Source: ThingSpeak Channel 3385868

#import Machine Learning library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import time

# Who Standards
WHO = {
    'pH':          {'min': 6.5,  'max': 8.5,   'unit': ''},
    'Turbidity':   {'min': 0.0,  'max': 4.0,   'unit': 'NTU'},
    'Temperature': {'min': 10.0, 'max': 35.0,  'unit': '°C'},
    'EC':          {'min': 0.0,  'max': 1500.0, 'unit': 'µS/cm'},
    'DO':          {'min': 4.0,  'max': 20.0,  'unit': 'mg/L'},
    'DHT_Temp':    {'min': 0.0,  'max': 50.0,  'unit': '°C'},
}

FEATURES = ['pH', 'Turbidity', 'Temperature', 'EC', 'DO', 'DHT_Temp']
CONFIDENCE_THRESHOLD = 70.0

# Fetch data from ThingSpeak
CHANNEL_ID   = "3385868"
READ_API_KEY = "TE478YBUM6UK5SUU"
WRITE_API_KEY = "OEVI320AAJ8KW1SL"
INTERVAL_SEC  = 15
RESULTS      = 100

print("=" * 60)
print("  Smart Water Quality Monitoring — Random Forest ")
print("  Khartoum State, Sudan | ThingSpeak Live Data")
print("=" * 60)
print("\n Fetching data from ThingSpeak...")

url = (f"https://api.thingspeak.com/channels/{CHANNEL_ID}/"
       f"feeds.json?api_key={READ_API_KEY}&results={RESULTS}")

response = requests.get(url, timeout=15)
if response.status_code != 200:
    raise ConnectionError(f"Connection failed {response.status_code}")

feeds = response.json().get('feeds', [])
if not feeds:
    raise ValueError("no data in channel!")

df_raw = pd.DataFrame(feeds)
df = pd.DataFrame({
    'pH':          pd.to_numeric(df_raw['field1'], errors='coerce'),
    'Turbidity':   pd.to_numeric(df_raw['field2'], errors='coerce'),
    'Temperature': pd.to_numeric(df_raw['field3'], errors='coerce'),
    'EC':          pd.to_numeric(df_raw['field4'], errors='coerce'),
    'DO':          pd.to_numeric(df_raw['field5'], errors='coerce'),
    'DHT_Temp':    pd.to_numeric(df_raw['field6'], errors='coerce'),
})
df = df.dropna().reset_index(drop=True)
print(f" fetched successfully {len(df)} read ✅")

# Write function to ThingSpeak
# Field 7 → Confidence (0-100)
# Field 8 → Status (0=Safe | 1=Danger | 2=Suspicious)

def write_to_thingspeak(confidence, status_code):
    url    = "https://api.thingspeak.com/update"
    params = {
        'api_key': WRITE_API_KEY,
        'field7':  round(confidence, 1),
        'field8':  int(status_code),
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.text.strip() == '0':
            print("ThangSpeak refused to write — less than 15 seconds between writes")
            return False
        print(
            f"✅ ThingSpeak ← "
            f"Field7(Confidence)={confidence:.1f}% | "
            f"Field8(Status)={status_code} "
            f"({'SAFE' if status_code==0 else 'UNSAFE' if status_code==1 else 'SUSPECT'})"
        )
        return True
    except Exception as e:
        print(f"Typing error: {e}")
        return False
  

# Classify Water to get the expect data
def classify_water(row):
    for feat in ['pH', 'Turbidity', 'Temperature', 'EC', 'DO']:
        if row[feat] < WHO[feat]['min'] or row[feat] > WHO[feat]['max']:
            return 1
    return 0

df['Label'] = df.apply(classify_water, axis=1)
n_normal   = (df['Label'] == 0).sum()
n_abnormal = (df['Label'] == 1).sum()
print(f"Normal: {n_normal} | Abnormal: {n_abnormal}")

# Adding abnormal data close to(Border-Zone Augmentation)
if n_abnormal < 10:
    print("\n⚠️  Apply Border-Zone Augmentation...")
    np.random.seed(42)
    n_synth = max(80, n_normal)

    synth = pd.DataFrame({
        'pH': np.concatenate([
            np.random.uniform(6.0,  6.49, n_synth // 2),
            np.random.uniform(8.51, 9.5,  n_synth // 2)]),
        'Turbidity': np.random.uniform(4.01, 7.0, n_synth),
        'Temperature': np.concatenate([
            np.random.uniform(7.0,  9.9,  n_synth // 2),
            np.random.uniform(35.1, 38.0, n_synth // 2)]),
        'EC':    np.random.uniform(1501, 2000, n_synth),
        'DO':    np.random.uniform(2.0,  3.99, n_synth),
        'DHT_Temp': np.random.uniform(10.0, 45.0, n_synth),
        'Label': np.ones(n_synth, dtype=int),
    })
    df = pd.concat([df, synth], ignore_index=True)
    n_normal   = (df['Label'] == 0).sum()
    n_abnormal = (df['Label'] == 1).sum()
    print(f"✅ after  adding — Normal: {n_normal} | Abormal: {n_abnormal}")

# Training Random Forest
X = df[FEATURES].values
y = df['Label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("\n Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_s, y_train)

y_pred = rf.predict(X_test_s)
acc    = accuracy_score(y_test, y_pred) * 100
print(f"✅ accuracy Random Forest: {acc:.2f}%")
print(classification_report(y_test, y_pred,
      target_names=['Normal', 'Abnormal']))

#  Decision Engine
def get_violations(reading_dict):
    """ Features that have exceeded the limits of WHO"""
    viols = []
    for feat in ['pH', 'Turbidity', 'Temperature', 'EC', 'DO']:
        val = reading_dict[feat]
        w   = WHO[feat]
        if val < w['min']:
            viols.append(
                f"{feat}: {val:.3f} {w['unit']} "
                f"<  WHO Minimum ({w['min']})")
        elif val > w['max']:
            viols.append(
                f"{feat}: {val:.3f} {w['unit']} "
                f">  WHO Maximum ({w['max']})")
    return viols

def hybrid_decision(reading_list, rf_model, scl):
    """
     Rule-Based + Random Forest
    """
    reading_dict = dict(zip(FEATURES, reading_list))

    #  Rule-Based
    violations = get_violations(reading_dict)
    if violations:
        return {
            'status':     '⚠️  ALARM — UNSAFE',
            'confidence': 100.0,
            'layer':      'Rule-Based',
            'color':      'red',
            'violations': violations,
        }

    #  Random Forest
    scaled = scl.transform([reading_list])
    pred   = rf_model.predict(scaled)[0]
    probs  = rf_model.predict_proba(scaled)[0]
    conf   = max(probs) * 100

    if conf < CONFIDENCE_THRESHOLD:
        return {
            'status':     '🟡 SUSPECT — UNCERTAIN',
            'confidence': conf,
            'layer':      'RF-Uncertain',
            'color':      'orange',
            'violations': ['Model confidence is low — manual review is recommended'],
        }

    if pred == 1:
        return {
            'status':     '⚠️  ALARM — UNSAFE',
            'confidence': conf,
            'layer':      'Random Forest',
            'color':      'red',
            'violations': ['Random Forest Discover an abnormal pattern '],
        }

    return {
        'status':     '✅ SAFE — Normal',
        'confidence': conf,
        'layer':      'RF + Rules',
        'color':      'green',
        'violations': [],
    }

#  Gray and critical case testing
print("\n" + "=" * 60)
print("Gray and critical case testing")
print("=" * 60)

test_cases = [
    ([8.48, 1.50, 24.0,  500.0, 6.5, 25.0], "Safe on the edge (pH=8.48)"),
    ([8.53, 1.20, 23.5,  480.0, 6.8, 24.0], "Unsafe (pH=8.53 > 8.5)"),
    ([7.20, 3.91, 26.0,  400.0, 6.0, 26.5], "Safe on the edge (Turb=3.91)"),
    ([7.40, 4.15, 25.8,  450.0, 5.8, 26.0], "Unsafe(Turb=4.15 > 4.0)"),
    ([8.45, 3.85, 34.0, 1450.0, 5.1, 33.0], "Gray area (all transactions are on the edge)"),
]

print(f"\n{'pH':>6} {'Turb':>6} {'Temp':>6} {'EC':>7} {'DO':>5} "
     f"{'Decision':>22} {'Trust':>8} {'Class':>15}")
print("─" * 82)

for reading, desc in test_cases:
    result = hybrid_decision(reading, rf, scaler)
    print(f"{reading[0]:>6.2f} {reading[1]:>6.2f} {reading[2]:>6.2f} "
          f"{reading[3]:>7.1f} {reading[4]:>5.2f} "
          f"{result['status']:>22} {result['confidence']:>7.1f}% "
          f"{result['layer']:>15}")
    print(f"       ↳ {desc}")
    if result['violations']:
        for v in result['violations']:
            print(f"         • {v}")
    print()


#  Analysis of the latest real reading from ThingSpeak
print("\n" + "=" * 60)
print("Starting real-time monitoring every 15 seconds...")
print("Press Ctrl+C to stop")
print("=" * 60)

last_write = time.time()

while True:
    try:
        
        # Fetching the latest 10 readings to filter the results and access the latest sensor reading
        url_live = (
            f"https://api.thingspeak.com/channels/{CHANNEL_ID}/"
            f"feeds.json?api_key={READ_API_KEY}&results=10"
        )
        resp = requests.get(url_live, timeout=10)
        feeds = resp.json().get('feeds', [])

        #  the latest reading that contains sensor data (where field1 is not None)
        feed = None
        for f in reversed(feeds):
            if f.get('field1') is not None:
                feed = f
                break

        if feed is None:
            feed = feeds[-1] if feeds else {}

        reading = [
            float(feed.get('field1') or 0),
            float(feed.get('field2') or 0),
            float(feed.get('field3') or 0),
            float(feed.get('field4') or 0),
            float(feed.get('field5') or 0),
            float(feed.get('field6') or 0),
        ]

        # Analyzing the reading 
        result = hybrid_decision(reading, rf, scaler)
        ts     = time.strftime('%H:%M:%S')

        # Printing the result
        print(f"\n[{ts}] ----------------------------------------")
        print(f"  pH:          {reading[0]:.3f} {WHO['pH']['unit']}")
        print(f"  Turbidity:   {reading[1]:.3f} {WHO['Turbidity']['unit']}")
        print(f"  Temperature: {reading[2]:.3f} {WHO['Temperature']['unit']}")
        print(f"  EC:          {reading[3]:.3f} {WHO['EC']['unit']}")
        print(f"  DO:          {reading[4]:.3f} {WHO['DO']['unit']}")
        print(f"  DHT Temp:    {reading[5]:.3f} {WHO['DHT_Temp']['unit']}")
        print(f"\n  {result['status']} ({result['confidence']:.1f}%)")
        print(f" Decision via: {result['layer']}")
        if result['violations']:
            print("  violations:")
            for v in result['violations']:
                print(f"    • {v}")

        #Sending to ThingSpeak
        status_map = {
            '⚠️  ALARM — UNSAFE':    1,
            '🟡 SUSPECT — UNCERTAIN': 2,
            '✅ SAFE — Normal':       0,
        }
        elapsed = time.time() - last_write
        if elapsed >= INTERVAL_SEC:
            status_code = status_map.get(result['status'], 0)
            write_to_thingspeak(result['confidence'], status_code)
            last_write = time.time()

        time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n ⏹ Monitoring stopped.")
        break
    except Exception as e:
        print(f"❌ Error: {e} — retrying...")
        time.sleep(INTERVAL_SEC)

