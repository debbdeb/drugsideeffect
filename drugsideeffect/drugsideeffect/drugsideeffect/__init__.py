# __init__.py

# Import main processing function
from .processing import main, process_onset_data, classify_sideeffects, clean_text, extract_symptoms, normalize_slang, extract_known_symptoms, extract_uncommon_symptoms, load_and_process_data

# Import all visualization functions
from .visualization import (
    plot_day_of_week_distribution,
    plot_data_count_per_month,
    plot_sentiment_distribution,
    plot_known_symptoms,
    plot_uncommon_side_effects_pie_chart,
    plot_proportion_of_english_words,
    plot_create_side_effects_correlation,
    plot_create_side_effects_visualizations,
    plot_visualize,
    plot_symptom_extraction,
    plot_onset_times
)

# Package version
__version__ = "0.1.0"

# Define all accessible names
__all__ = [
    "main",
    "process_onset_data",
    "classify_sideeffects",
    "clean_text",
    "extract_symptoms",
    "normalize_slang",
    "extract_known_symptoms",
    "extract_uncommon_symptoms",
    "load_and_process_data",
    "plot_day_of_week_distribution",
    "plot_data_count_per_month",
    "plot_sentiment_distribution",
    "plot_known_symptoms",
    "plot_uncommon_side_effects_pie_chart",
    "plot_proportion_of_english_words",
    "plot_create_side_effects_correlation",
    "plot_create_side_effects_visualizations",
    "plot_visualize",
    "plot_symptom_extraction",
    "plot_onset_times"
]

# -----------------------------
# Pipeline function
# -----------------------------
def sideeffect(input_file_path):
    """
    Run the full sideeffect analysis and visualization pipeline.
    This includes:
    - Side effect classification
    - Text cleaning
    - Symptom extraction (known/uncommon)
    - Alert keyword detection
    - All plotting functions
    Returns the final DataFrame.
    """
    # Step 1: Load and process CSV
    df = main(input_file_path)

    # Step 2: Basic distributions
    plot_day_of_week_distribution(df)
    plot_data_count_per_month(df)
    plot_sentiment_distribution(df)

    # Step 3: Symptom plots
    plot_known_symptoms(df)
    plot_uncommon_side_effects_pie_chart(df)
    plot_proportion_of_english_words(df)
    plot_symptom_extraction(df)

    # Step 4: Correlation & detailed visualizations
    plot_create_side_effects_correlation(df)
    plot_create_side_effects_visualizations(df)
    plot_visualize(df)

    # Step 5: Onset time plots
    if 'Date' in df.columns:
        df = process_onset_data(df)
        plot_onset_times(df)

    return df
