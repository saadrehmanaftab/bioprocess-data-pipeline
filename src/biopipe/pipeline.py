"""Ingest raw site exports, harmonize and clean them with a data quality log, load a SQLite data model,
build early process features and train a scikit-learn model that predicts the final titer."""
import pathlib, re, sqlite3
import numpy as np
import pandas as pd

MAP_B_ONLINE = dict(Time_h="hour", Temp_F="temp", pH_value="ph", DO_pct="do")
MAP_B_OFFLINE = dict(Day="day", VCD_e6_per_mL="vcd", Viab_pct="viability", Glc_mmolL="glucose", Titer_gL="titer")
LIMITS = dict(temp=(30, 40), ph=(6.0, 8.0), do=(0, 100))

def harmonize(df, kind, site):
    if site == "B":
        df = df.rename(columns=MAP_B_ONLINE if kind == "online" else MAP_B_OFFLINE)
        if kind == "online": df["temp"] = (df.temp - 32) * 5 / 9
        else: df["glucose"] = df.glucose * 0.18016
    return df

def clean_online(df, log, bid):
    n0 = len(df); df = df.drop_duplicates(subset="hour").sort_values("hour")
    log.append((bid, "online", "duplicate rows removed", n0 - len(df)))
    for col, (lo, hi) in LIMITS.items():
        bad = (df[col] < lo) | (df[col] > hi)
        log.append((bid, "online", f"{col} out of range set to missing", int(bad.sum())))
        df.loc[bad, col] = np.nan
    miss = int(df[list(LIMITS)].isna().sum().sum())
    df[list(LIMITS)] = df[list(LIMITS)].interpolate(limit=3, limit_direction="both")
    log.append((bid, "online", "gaps up to 3 h interpolated", miss - int(df[list(LIMITS)].isna().sum().sum())))
    return df

def ingest(raw):
    raw = pathlib.Path(raw); log = []; on_all, off_all = [], []
    for f in sorted(raw.glob("site*_batch*_online.csv")):
        site, bid = re.match(r"site(\w)_batch(\d+)_online", f.stem).groups(); bid = int(bid)
        on = clean_online(harmonize(pd.read_csv(f), "online", site), log, bid); on["batch"] = bid
        off = harmonize(pd.read_csv(raw / f.name.replace("online", "offline")), "offline", site); off["batch"] = bid
        on_all.append(on); off_all.append(off)
    meta = pd.read_csv(raw / "batch_setpoints.csv")
    qlog = pd.DataFrame(log, columns=["batch", "table", "check", "count"])
    return meta, pd.concat(on_all, ignore_index=True), pd.concat(off_all, ignore_index=True), qlog

def load_sqlite(path, meta, online, offline, qlog):
    path = pathlib.Path(path)
    if path.exists(): path.unlink()
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE batches(batch INTEGER PRIMARY KEY, site TEXT, temp_set REAL, ph_set REAL, do_set REAL, feed REAL);
    CREATE TABLE online(batch INTEGER REFERENCES batches(batch), hour INTEGER, temp REAL, ph REAL, do REAL,
                        PRIMARY KEY(batch, hour));
    CREATE TABLE offline(batch INTEGER REFERENCES batches(batch), day INTEGER, vcd REAL, viability REAL,
                         glucose REAL, titer REAL, PRIMARY KEY(batch, day));
    CREATE TABLE quality_log(batch INTEGER, tbl TEXT, check_name TEXT, count INTEGER);
    """)
    con.executemany("INSERT INTO batches VALUES (?,?,?,?,?,?)", meta[["batch", "site", "temp_set", "ph_set", "do_set", "feed"]].values.tolist())
    con.executemany("INSERT INTO online VALUES (?,?,?,?,?)", online[["batch", "hour", "temp", "ph", "do"]].values.tolist())
    con.executemany("INSERT INTO offline VALUES (?,?,?,?,?,?)", offline[["batch", "day", "vcd", "viability", "glucose", "titer"]].values.tolist())
    con.executemany("INSERT INTO quality_log VALUES (?,?,?,?)", qlog.values.tolist())
    con.commit(); return con

def features(con, upto_day=5):
    q = f"""
    SELECT b.batch, b.site,
           AVG(o.temp) AS temp_mean, AVG(o.ph) AS ph_mean, AVG(o.do) AS do_mean,
           (SELECT vcd FROM offline f WHERE f.batch=b.batch AND f.day={upto_day}) AS vcd_d{upto_day},
           (SELECT vcd FROM offline f WHERE f.batch=b.batch AND f.day=2) AS vcd_d2,
           (SELECT AVG(glucose) FROM offline f WHERE f.batch=b.batch AND f.day<={upto_day}) AS glc_mean,
           (SELECT titer FROM offline f WHERE f.batch=b.batch AND f.day=14) AS titer_final
    FROM batches b JOIN online o ON o.batch=b.batch AND o.hour<={upto_day * 24}
    GROUP BY b.batch"""
    return pd.read_sql_query(q, con)

def train(X, y, seed=0):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = KFold(5, shuffle=True, random_state=seed); out = {}
    models = dict(ridge=make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
                  random_forest=RandomForestRegressor(300, min_samples_leaf=2, random_state=seed))
    for name, mdl in models.items():
        p = cross_val_predict(mdl, X, y, cv=cv)
        out[name] = dict(mae_gL=float(np.mean(np.abs(p - y))), mape_pct=float(100 * np.mean(np.abs(p - y) / y)),
                         r2=float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)))
    out["mean_baseline"] = dict(mape_pct=float(100 * np.mean(np.abs(y.mean() - y) / y)))
    return out
