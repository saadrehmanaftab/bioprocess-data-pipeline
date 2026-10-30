import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from biopipe import generate, pipeline

ROOT = pathlib.Path(__file__).parent
raw = generate.write_raw(ROOT / "data" / "raw")
meta, on, off, qlog = pipeline.ingest(raw)
con = pipeline.load_sqlite(ROOT / "results" / "bioprocess.sqlite", meta, on, off, qlog)
feat = pipeline.features(con)
cols = ["temp_mean", "ph_mean", "do_mean", "vcd_d5", "vcd_d2", "glc_mean"]
res = pipeline.train(feat[cols].values, feat.titer_final.values)
q = qlog.groupby("check")["count"].sum().to_dict()
summary = dict(batches=int(len(meta)), sites=2, online_rows=int(len(on)), offline_rows=int(len(off)),
               quality_fixes=q, quality_fixes_total=int(sum(q.values())), model_day5_features=res)
feat.to_csv(ROOT / "results" / "features_day5.csv", index=False)
(ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=1))
