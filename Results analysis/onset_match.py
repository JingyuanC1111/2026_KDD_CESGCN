import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.signal import argrelextrema
import ace_tools as tools

# ------------------------------
# 1. Load the two uploaded files
# ------------------------------
hosp = pd.read_csv("/mnt/data/hospital_admissions_state_abbr.csv", parse_dates=["date"])
hosp.set_index("date", inplace=True)

mte = pd.read_csv("/mnt/data/all_pairs_first_7_weeks.csv", parse_dates=["date"])
mte['source_state'] = mte['source_state'].astype(str)
mte['target_state'] = mte['target_state'].astype(str)

# --------------------------------------------------------
# 2. Prep: smooth hospitalisations with 3‑week rolling mean
# --------------------------------------------------------
hosp_smooth = hosp.rolling(window=3, center=True, min_periods=1).mean()


# --------------------------------------------------------
# 3. Define “take‑off” = first week (within last‑12 window)
#    where:
#      • current value >= 5
#      • values rise monotonically for 2 consecutive weeks
# --------------------------------------------------------
def onset_in_last_12(series, thresh=5, rise_weeks=2):
    """
    Returns index (0‑based within 12‑week slice) of first take‑off week.
    If none found, returns None.
    """
    for idx in range(rise_weeks - 1, len(series)):
        win = series[(idx - (rise_weeks - 1)):(idx + 1)]
        if win.is_monotonic_increasing and win.iloc[-1] >= thresh:
            return idx
    return None


# --------------------------------------------------------
# 4. Validate every MTE pair with 12‑week window
# --------------------------------------------------------
records = []
for _, row in mte.iterrows():
    det = row['date']
    lag_mte = int(row['lag'])
    src, tgt = row['source_state'], row['target_state']

    if det not in hosp_smooth.index:  # ensure date exists
        continue
    if src not in hosp_smooth.columns or tgt not in hosp_smooth.columns:
        continue

    window = hosp_smooth.loc[:det].tail(12)
    if len(window) < 12:  # need full 12 weeks backward
        continue

    src_series = window[src]
    tgt_series = window[tgt]

    src_onset_idx = onset_in_last_12(src_series)
    tgt_onset_idx = onset_in_last_12(tgt_series)

    if src_onset_idx is None or tgt_onset_idx is None:
        continue

    lag_obs = tgt_onset_idx - src_onset_idx  # difference in weeks (0‑based)

    records.append({
        "date": det.date(),
        "source": src,
        "target": tgt,
        "lag_mte": lag_mte,
        "lag_obs": lag_obs,
        "match_exact": lag_obs == lag_mte,
        "match_loose": abs(lag_obs - lag_mte) <= 1
    })

df_verify = pd.DataFrame(records)

# ------------------------------
# 5. Display results
# ------------------------------
tools.display_dataframe_to_user("MTE Validation – 12‑Week Onset Check", df_verify)

summary = pd.DataFrame({
    "Result": ["Exact match", "Loose match (±1)", "Mismatch / No onset"],
    "Count": [
        df_verify['match_exact'].sum(),
        df_verify['match_loose'].sum(),
        len(mte) - df_verify['match_loose'].sum()
    ]
})
tools.display_dataframe_to_user("Summary – 12‑Week Onset Check", summary)
