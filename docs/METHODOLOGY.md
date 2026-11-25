# Methodology summary

The task is one-hour-ahead prediction of hourly PM2.5 concentration. The archived experiment compared persistence, 24-hour seasonal-naive, and Random Forest models across six chronological expanding-window origins, each with a 30-day test period.

The dataset contains 43,824 hourly rows from 2010-01-01 00:00 to 2014-12-31 23:00. PM2.5 is missing for 2,067 rows (4.717%). The original analysis did not impute missing target values for evaluation.

Availability mapping and reporting-delay sensitivity are assumption-based because the dataset does not include actual sensor publication/ingestion timestamps. The one-hour delay scenario is a sensitivity assumption, not a measured operational delay.
