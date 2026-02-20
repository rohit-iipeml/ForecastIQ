#!/bin/bash

# Run forecast for 00Z initialization
python src/gru_universal.py 2025-01-25 00 \
  --history_csv "data/data_history.csv" \
  --weather_dir "data/weather_daily" \
  --load_dir "data/daily_load" \
  --out_dir "outputs"

# Run forecast for 12Z initialization
python src/gru_universal.py 2025-01-24 12 \
  --history_csv "data/data_history12.csv" \
  --weather_dir_12 "data/weather_daily_12" \
  --load_dir "data/daily_load" \
  --out_dir "outputs"