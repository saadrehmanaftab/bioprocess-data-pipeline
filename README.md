# Bioprocess lab data pipeline (personal project, synthetic data)

Personal project. A small data engineering and data science pipeline for bioreactor batch data: raw
exports from two lab sites are harmonized, cleaned with a data quality log, loaded into a SQLite data
model, turned into early process features and used to predict the final titer with scikit-learn.

**All data are synthetic** (`src/biopipe/generate.py`): an assumed growth and titer model for 60 fed
batch runs of 14 days, with assumed defects. No real process or company data are used.

## Pipeline (`src/biopipe/pipeline.py`, run with `python run_pipeline.py`)
1. **Ingest and harmonize:** site B exports deg F, glucose in mmol/L and different column names; all
   are mapped to one schema in deg C and g/L.
2. **Data quality:** duplicate timestamps removed, physically impossible sensor values set to missing,
   gaps up to 3 h interpolated; every fix is written to a `quality_log` table.
3. **Data model (SQLite):** `batches`, `online` (hourly pH, DO, temperature), `offline` (daily VCD,
   viability, glucose, titer) and `quality_log`, with primary and foreign keys.
4. **Features in SQL:** process means and cell density up to day 5.
5. **Model:** ridge regression and random forest, 5 fold cross validation, against a mean baseline.

## Result (`results/summary.json`)
- 60 batches, 20,220 online rows and 900 offline rows loaded; 786 data quality fixes logged
  (180 duplicates, 101 pH spikes, 505 interpolated gap values).
- Final titer from day 5 data: ridge regression 8.1% mean absolute percentage error (R2 0.83),
  random forest 15.9%, mean baseline 23.5%.

## Limits
Synthetic data from a simple assumed model, no real lab system or cloud storage, small data set, no
hyperparameter search beyond the ridge alpha.

## Run
```
pip install numpy pandas scikit-learn pytest
python run_pipeline.py
pytest -q
```
