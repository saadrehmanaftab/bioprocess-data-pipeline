import sys, pathlib, sqlite3
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from biopipe import generate, pipeline

def test_site_b_units_are_harmonized():
    on = pd.DataFrame(dict(Time_h=[0], Temp_F=[98.6], pH_value=[7.0], DO_pct=[40.0]))
    off = pd.DataFrame(dict(Day=[0], VCD_e6_per_mL=[1.0], Viab_pct=[99.0], Glc_mmolL=[33.3], Titer_gL=[0.0]))
    assert abs(pipeline.harmonize(on, "online", "B").temp[0] - 37.0) < 1e-9
    assert abs(pipeline.harmonize(off, "offline", "B").glucose[0] - 33.3 * 0.18016) < 1e-12

def test_cleaning_removes_duplicates_and_spikes():
    df = pd.DataFrame(dict(hour=[0, 1, 1, 2, 3], temp=[37.0] * 5, ph=[7.0, 14.0, 14.0, 7.0, 7.0], do=[40.0] * 5))
    log = []; out = pipeline.clean_online(df, log, 1)
    assert len(out) == 4 and out.ph.between(6, 8).all()

def test_pipeline_end_to_end(tmp_path):
    raw = generate.write_raw(tmp_path / "raw", n=12, seed=1)
    meta, on, off, q = pipeline.ingest(raw)
    con = pipeline.load_sqlite(tmp_path / "db.sqlite", meta, on, off, q)
    n = con.execute("SELECT COUNT(*) FROM online o LEFT JOIN batches b USING(batch) WHERE b.batch IS NULL").fetchone()[0]
    assert n == 0                                                   # every row belongs to a batch
    assert con.execute("SELECT COUNT(*) FROM online WHERE ph<6 OR ph>8").fetchone()[0] == 0
    f = pipeline.features(con); assert len(f) == 12 and f.titer_final.notna().all()
