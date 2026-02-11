def sideeffect_pipeline(
    csv_file,
    project_root=r"C:\Users\91990\Desktop",
    hide_warnings=True,
    hide_nltk_messages=True
):
    """
    Runs the full SideEffect visualization pipeline.
    - Uses trained PKL model + vectorizer
    - Prints initial and filtered row counts
    - Generates plots
    """

    # -----------------------------
    # Setup
    # -----------------------------
    import os
    import sys
    import warnings
    import pandas as pd
    import matplotlib.pyplot as plt

    if hide_warnings:
        warnings.filterwarnings("ignore")

    # Ensure project is importable
    os.chdir(project_root)
    sys.path.append(os.getcwd())

    # -----------------------------
    # Silence NLTK messages
    # -----------------------------
    if hide_nltk_messages:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("words", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    # -----------------------------
    # Imports (after path setup)
    # -----------------------------
    from drugsideeffect.processing import main, process_onset_data
    from drugsideeffect.visualization import (
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
        plot_onset_times,
    )

    # -----------------------------
    # Load & classify data
    # -----------------------------
    df_initial = pd.read_csv(csv_file)
    print(f"Initial CSV rows: {len(df_initial)}")

    df = main(csv_file)
    print(f"Rows after side effect filtering: {len(df)}")

    # -----------------------------
    # Onset / duration processing
    # -----------------------------
    df = process_onset_data(df)

    # -----------------------------
    # ALL PLOTS (none deleted)
    # -----------------------------
    plot_symptom_extraction(df); plt.show()
    plot_day_of_week_distribution(df); plt.show()
    plot_data_count_per_month(df); plt.show()
    plot_sentiment_distribution(df); plt.show()
    plot_create_side_effects_correlation(df); plt.show()
    plot_create_side_effects_visualizations(df); plt.show()
    plot_visualize(df); plt.show()
    plot_known_symptoms(df); plt.show()
    plot_uncommon_side_effects_pie_chart(df); plt.show()
    plot_proportion_of_english_words(df); plt.show()
    plot_onset_times(df); plt.show()

    print("All plots generated successfully!")

    return df
