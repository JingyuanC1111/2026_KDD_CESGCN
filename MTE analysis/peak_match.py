# Define function to detect global peak (index of maximum value in last 12 weeks)
def global_peak(series):
    return series.idxmax()

# Redo validation using global peak instead of onset
records_peak = []
for _, row in mte.iterrows():
    det = row['date']
    lag_mte = int(row['lag'])
    src, tgt = row['source_state'], row['target_state']

    if det not in hosp_smooth.index:
        continue
    if src not in hosp_smooth.columns or tgt not in hosp_smooth.columns:
        continue

    window = hosp_smooth.loc[:det].tail(12)
    if len(window) < 12:
        continue

    src_series = window[src]
    tgt_series = window[tgt]

    try:
        src_peak_date = global_peak(src_series)
        tgt_peak_date = global_peak(tgt_series)
    except:
        continue

    lag_obs = (tgt_peak_date - src_peak_date).days // 7

    records_peak.append({
        "date": det.date(),
        "source": src,
        "target": tgt,
        "lag_mte": lag_mte,
        "src_peak": src_peak_date.date(),
        "tgt_peak": tgt_peak_date.date(),
        "lag_obs": lag_obs,
        "match_exact": lag_obs == lag_mte,
        "match_loose": abs(lag_obs - lag_mte) <= 1
    })

df_global_peak = pd.DataFrame(records_peak)

# Display full validation table
tools.display_dataframe_to_user("MTE Validation – Global Peak (12-week)", df_global_peak)

# Create summary
summary_peak = pd.DataFrame({
    "Result": ["Exact match", "Loose match (±1)", "Mismatch / No peak"],
    "Count": [
        df_global_peak['match_exact'].sum(),
        df_global_peak['match_loose'].sum(),
        len(mte) - df_global_peak['match_loose'].sum()
    ]
})
tools.display_dataframe_to_user("Summary – Global Peak Check", summary_peak)
