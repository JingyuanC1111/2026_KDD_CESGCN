# ------------------------
# Code snippet: strict validation & plotting
# ------------------------

import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# ---------- 1. Load data ----------
hosp = pd.read_csv("/mnt/data/hospital_admissions_state_abbr.csv", parse_dates=["date"])
hosp.set_index("date", inplace=True)
hosp_smooth = hosp.rolling(window=3, center=True, min_periods=1).mean()

mte = pd.read_csv("/mnt/data/MTE_Validations.csv", parse_dates=["date"])
mte['source'] = mte['source'].astype(str)
mte['target'] = mte['target'].astype(str)

# ---------- 2. Strict validation (fixed 4‑week forecast window) ----------
records = []
fixed_window = 4          # forecast window length in weeks
horizon_weeks = 8         # how far ahead to search for global peak

for _, row in mte.iterrows():
    det = row['date']
    lag_mte = int(row['lag_mte'])
    src, tgt = row['source'], row['target']

    if det not in hosp_smooth.index:
        continue
    if src not in hosp_smooth.columns or tgt not in hosp_smooth.columns:
        continue

    # Forecast window [t+1, t+4]
    fw_start = det + timedelta(weeks=1)
    fw_end = det + timedelta(weeks=fixed_window)
    forecast_window = hosp_smooth.loc[fw_start:fw_end, tgt]

    # Horizon window to determine global peak
    horizon_end = det + timedelta(weeks=horizon_weeks)
    horizon_window = hosp_smooth.loc[fw_start:horizon_end, tgt]

    if forecast_window.empty or horizon_window.empty:
        continue

    global_peak_date = horizon_window.idxmax()
    is_valid = fw_start <= global_peak_date <= fw_end

    records.append({
        "date": det.date(),
        "source": src,
        "target": tgt,
        "lag_mte": lag_mte,
        "forecast_start": fw_start.date(),
        "forecast_end": fw_end.date(),
        "global_peak_date": global_peak_date.date(),
        "valid_strict": is_valid
    })

df_strict = pd.DataFrame(records)
print(df_strict.head())    # quick preview

# ---------- 3. Plot utility ----------
def plot_pair(source, target, det_date, lag_mte=4):
    """
    Visualize source & target curves:
    - 12 wks before detection
    - 4 wks after detection
    - shaded forecast window (1..lag_mte)
    """
    det = pd.to_datetime(det_date)
    plot_start = det - timedelta(weeks=12)
    plot_end = det + timedelta(weeks=lag_mte)

    plot_df = hosp_smooth.loc[plot_start:plot_end, [source, target]]

    # Peak of target in horizon for annotation
    horizon_end = det + timedelta(weeks=8)
    tgt_peak_date = hosp_smooth.loc[det + timedelta(weeks=1):horizon_end, target].idxmax()
    tgt_peak_val = hosp_smooth.loc[tgt_peak_date, target]

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df.index, plot_df[source], label=f"{source} (source)", linewidth=2)
    plt.plot(plot_df.index, plot_df[target], label=f"{target} (target)", linewidth=2)

    plt.axvline(det, color="black", linestyle="--", label="Detection")
    plt.axvspan(det + timedelta(weeks=1),
                det + timedelta(weeks=lag_mte),
                color="gray", alpha=0.25, label="Forecast Window (4 wks)")
    plt.scatter(tgt_peak_date, tgt_peak_val, color="purple", zorder=5, label=f"{target} Peak")

    plt.title(f"{source} → {target} | Detection: {det.date()} | MTE lag={lag_mte}")
    plt.xlabel("Date"); plt.ylabel("Smoothed Admissions")
    plt.xticks(rotation=45); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

# ---------- 4. Example plot: CA → NJ ----------
# Find one CA→NJ detection entry
example = mte[(mte["source"] == "CA") & (mte["target"] == "NJ")].iloc[0]
plot_pair("CA", "NJ", example['date'], lag_mte=4)
