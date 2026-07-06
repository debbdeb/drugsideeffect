# DrugSideEffect

**DrugSideEffect** is a Python package for exploratory pharmacovigilance that extracts, analyzes, and visualizes potential drug side-effect signals from unstructured text such as social media posts, surveys, and patient reports.

It provides an end-to-end NLP pipeline for:
- Text preprocessing and cleaning  
- Symptom and side-effect extraction (known + uncommon + slang-based)  
- Alert keyword detection  
- Sentiment analysis of health-related narratives  
- Temporal and distribution-based analysis  
- Visualization of pharmacovigilance signals  

---

## Disclaimer

DrugSideEffect is intended solely for educational and research purposes. It is not a medical device and does not provide medical advice, diagnosis, or treatment. Results produced by this software should not be used for clinical decision-making. Always consult a qualified healthcare professional for medical concerns.

---

## Installation

```bash
pip install drugsideeffect


pypi repository link https://pypi.org/project/drugsideeffect/

Note: The code is provided for exploratory purposes; it is not fully optimized and generates basic visualizations.



# Quick Start Example

# ==========================================================
# DEMO CODE for  pypi package drugsideeffect
# ==========================================================

# ---- STEP 1: INSTALL PACKAGE (FROM OFFICIAL PYPI) ----
!pip install --upgrade drugsideeffect

# ---- STEP 2: IMPORT LIBRARIES ----
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# ---- STEP 3: LOAD DATA ----
try:
    df = pd.read_csv(r"C:\Users\91990\Desktop\data.csv")
    print("Dataset loaded successfully")
    print(df.head())
except Exception as e:
    print("Error loading dataset:", e)
    raise

# ---- STEP 4: IMPORT PACKAGE ----
try:
    from drugsideeffect.processing import main
    from drugsideeffect.visualization import (
        plot_symptom_extraction,
        plot_day_of_week_distribution,
        plot_data_count_per_month,
        plot_sentiment_distribution,
        plot_create_side_effects_correlation,
        plot_create_side_effects_visualizations,
        plot_visualize,
        plot_known_symptoms,
        plot_uncommon_side_effects_pie_chart,
        plot_proportion_of_english_words
    )
    print("Package imported successfully")

except Exception as e:
    print("Import error:", e)
    print("Fix: reinstall package or check installed version on PyPI")
    raise

# ---- STEP 5: FEATURE EXTRACTION ----
print("\n Running symptom extraction...")
plot_symptom_extraction(df)

# ---- STEP 6: BASIC ANALYSIS ----
print("\n Running basic analytics...")
plot_day_of_week_distribution(df)
plot_data_count_per_month(df)
plot_sentiment_distribution(df)

# ---- STEP 7: ADVANCED PHARMACOVIGILANCE ANALYSIS ----
print("\n Running pharmacovigilance visualization...")
plot_create_side_effects_correlation(df)
plot_create_side_effects_visualizations(df)
plot_visualize(df)

# ---- STEP 8: SYMPTOM + DATA QUALITY ANALYSIS ----
print("\n Running symptom & data quality analysis...")
plot_known_symptoms(df)
plot_uncommon_side_effects_pie_chart(df)
plot_proportion_of_english_words(df)

print("\nALL VISUALIZATION COMPLETED")