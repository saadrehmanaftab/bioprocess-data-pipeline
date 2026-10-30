"""Write SYNTHETIC raw exports of fed batch bioreactor runs from two lab sites, with typical defects.

ASSUMED: growth and titer model, process ranges, defect rates. Not real Sartorius or customer data.
Site A exports deg C and glucose in g/L; site B exports deg F, glucose in mmol/L and other column names.
"""
import pathlib
import numpy as np
import pandas as pd

def batch(rng, bid, site):
    T = rng.uniform(36.0, 37.5); pH = rng.uniform(6.85, 7.15); do = rng.uniform(30, 60); feed = rng.uniform(0.8, 1.2)
    mu = 0.035 * (1 - abs(T - 36.8) / 3) * (1 - abs(pH - 7.0) / 0.6) * (0.8 + 0.004 * do)  # 1/h, assumed
    hours = np.arange(0, 14 * 24 + 1)
    vcd = 0.4 * np.exp(mu * hours) / (1 + 0.4 * (np.exp(mu * hours) - 1) / (18 * feed))   # 1e6 cells/mL, logistic
    online = pd.DataFrame(dict(hour=hours, temp=T + rng.normal(0, 0.05, hours.size),
                               ph=pH + rng.normal(0, 0.01, hours.size), do=do + rng.normal(0, 2, hours.size)))
    days = np.arange(0, 15)
    v = np.interp(days * 24, hours, vcd)
    titer = 0.0009 * np.cumsum(v * 24) * (1.1 - 0.15 * abs(pH - 7.0) / 0.15) * feed      # g/L, assumed
    glc = np.clip(6 - 0.25 * np.cumsum(v) / feed + 3 * (days % 2 == 0), 0.5, None)       # g/L
    offline = pd.DataFrame(dict(day=days, vcd=v * rng.normal(1, 0.04, days.size),
                                viability=np.clip(99 - 0.02 * days ** 3 + rng.normal(0, 1, days.size), 50, 100),
                                glucose=glc, titer=titer * rng.normal(1, 0.03, days.size)))
    # defects
    drop = rng.random(hours.size) < 0.02; online.loc[drop, "do"] = np.nan                 # sensor dropouts
    spikes = rng.random(hours.size) < 0.005; online.loc[spikes, "ph"] = rng.choice([0.0, 14.0], spikes.sum())
    online = pd.concat([online, online.sample(frac=0.01, random_state=bid)]).sort_values("hour")   # duplicates
    if site == "B":
        online["temp"] = online.temp * 9 / 5 + 32
        offline["glucose"] = offline.glucose / 0.18016                                    # g/L to mmol/L
        online = online.rename(columns=dict(hour="Time_h", temp="Temp_F", ph="pH_value", do="DO_pct"))
        offline = offline.rename(columns=dict(day="Day", vcd="VCD_e6_per_mL", viability="Viab_pct",
                                              glucose="Glc_mmolL", titer="Titer_gL"))
    return online, offline, dict(batch=bid, site=site, temp_set=T, ph_set=pH, do_set=do, feed=feed)

def write_raw(folder, n=60, seed=5):
    folder = pathlib.Path(folder); folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed); meta = []
    for b in range(1, n + 1):
        site = "A" if b % 3 else "B"
        on, off, m = batch(rng, b, site)
        on.to_csv(folder / f"site{site}_batch{b:03d}_online.csv", index=False)
        off.to_csv(folder / f"site{site}_batch{b:03d}_offline.csv", index=False)
        meta.append(m)
    pd.DataFrame(meta).to_csv(folder / "batch_setpoints.csv", index=False)
    return folder
