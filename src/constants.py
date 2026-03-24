from __future__ import annotations

MIN_SAMPLES_RELIABLE_R2 = 30

# Stability percentage thresholds (fraction of median load)
STAB_PCT = {
    "moderate": {"disagree": 0.012, "vol": 0.008, "range": 0.020},
    "unstable": {"disagree": 0.025, "vol": 0.015, "range": 0.040},
    "highly": {"disagree": 0.040, "vol": 0.025, "range": 0.060},
}

# Capacity watchlist ratio boundaries
CAPACITY_AT_RATIO = 0.99
CAPACITY_NEAR_RATIO = 0.98

# Risk thresholds
RISK_SEVERE_EXCEED_MAJ = 3
RISK_SEVERE_MAX_EX_MW = 25
RISK_SEVERE_HRS_ABOVE = 8
RISK_HIGH_EXCEED_MAJ = 1
RISK_HIGH_MAX_EX_MW = 10
RISK_HIGH_HRS_ABOVE = 4
RISK_MEDIUM_MAX_EX_MIN = 3
RISK_LOW_MAX_EX_LT = 3
RISK_MEDIUM_HRS_MIN = 1
RISK_MEDIUM_HRS_MAX = 3

# Peak confidence thresholds
PEAK_STRONG_CONF = 0.8
PEAK_STRONG_SPREAD = 1
PEAK_OK_CONF = 0.6
PEAK_OK_SPREAD = 3

# Confidence grade and action thresholds
DISAGREEMENT_P50 = 1.8
DISAGREEMENT_P75 = 2.6
DISAGREEMENT_P90 = 3.8
GRADE_A_PEAK_CONF_MIN = 0.70
GRADE_A_SPREAD_MAX = 2
GRADE_B_PEAK_CONF_MIN = 0.55
GRADE_B_SPREAD_MAX = 4
GRADE_C_PEAK_CONF_MIN = 0.40
T2M_DISAGREEMENT_HIGH = 0.9
ATTRIBUTION_R2_HIGH = 0.25
REV_VOL_HIGH = 2.0
MAX_RANGE_HIGH = 8.0
VOI_MAJOR_IMPROVEMENT_MW = 2.0

# Ramp risk thresholds (fraction of avg load per hour)
RAMP_HIGH_PCT = 0.05   # 5% of avg load per hour triggers ramp risk flag

# Backtest quality thresholds (MAE as fraction of avg load)
BACKTEST_POOR_PCT = 0.03    # MAE > 3% of avg load → poor quality
BACKTEST_MODERATE_PCT = 0.015  # MAE 1.5-3% → moderate
