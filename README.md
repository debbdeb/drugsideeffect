# Python package sideeffect 

**sideeffect** is a Python library for processing textual health-related data to extract, analyze, and visualize potential side effects of pharmaceutical drugs from text sources such as social media posts, surveys, or reports.

It provides pipeline for:
- Cleaning and preprocessing text
- Extracting known and uncommon symptoms
- Detecting alert-related keywords
- Generating multiple analytical visualizations

---

# Disclaimer

**This library is only for informational, educational, and research purposes only.**  
It is **not intended to provide medical advice, diagnosis, or treatment**.  
Do **not** use it as a substitute for professional healthcare guidance.  
Always consult a qualified healthcare professional regarding any medical concerns.

Note: The code is provided for demonstration purposes; it is not fully optimized and generates only basic visualizations.

---

# Installation from PyPI:

%pip install sideeffect


# Quick Start Example

import warnings
warnings.filterwarnings("ignore")  # Hide warnings

import os
import pandas as pd

# Set working directory
os.chdir(r"C:\")

# Add the project folder to path
import sys
sys.path.append(r"C:\")

# Import the custom functions
from sideeffect.processing import main
from sideeffect.visualization import (
    plot_day_of_week_distribution,
    plot_data_count_per_month,
    plot_known_symptoms,
    plot_uncommon_side_effects_pie_chart,
    plot_proportion_of_english_words,
    plot_create_side_effects_correlation,
    plot_create_side_effects_visualizations,
    plot_sentiment_distribution,
    plot_visualize,
    plot_symptom_extraction,
)


# -----------------------------
# Step 1: Load a csv file having at least four columns for example:
## Headers: Date, month, RT_Like, text 
## Row values: 3/15/2023 8:23, 4, 1, Day 2 and I feel better


# -----------------------------
df = pd.read_csv("data.csv")


# -----------------------------
# Step 2: Extract symptoms & alert keywords
# -----------------------------
# This must be done first so that derived columns like 'extracted_symptoms' exist
plot_symptom_extraction(df)

# -----------------------------
# Step 3: Basic distributions & sentiment analysis
# -----------------------------
plot_day_of_week_distribution(df)
plot_data_count_per_month(df)
plot_sentiment_distribution(df)

# -----------------------------
# Step 4: Side effects correlation & detailed visualizations
# -----------------------------
plot_create_side_effects_correlation(df)
plot_create_side_effects_visualizations(df)
plot_visualize(df)

# -----------------------------
# Step 5: Plots that depend on extracted columns
# -----------------------------
plot_known_symptoms(df)
plot_uncommon_side_effects_pie_chart(df)
plot_proportion_of_english_words(df)

print("All plots generated successfully!")
