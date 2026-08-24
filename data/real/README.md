# Real data

This folder already contains the two datasets used in the paper (tracked in git):

- `Influenza2024.xlsx` — Irish 2024 influenza weekly case counts.
  Used by `real_data_study/flu_application.py`.
- `COVID-19_HPSC_Detailed_Statistics_Profile.csv` — Irish HPSC COVID-19
  detailed statistics. Used by `real_data_study/covid_application.py`.

Both applications read the file directly from this folder via a
`DATA_PATH` constant at the top of the script — update that constant if you
rename or move the files.
