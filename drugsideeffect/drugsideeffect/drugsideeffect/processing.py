# Processing.py

import pandas as pd
import numpy as np
import re
import os
from textblob import TextBlob
from datetime import datetime
import joblib

# -----------------------------
# Load pretrained artifacts
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "sideeffect_nb.pkl")
VEC_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

_model = joblib.load(MODEL_PATH)
_vectorizer = joblib.load(VEC_PATH)

# -----------------------------
# Side effect classification
# -----------------------------
def classify_sideeffects(df):
    """
    Adds a binary prediction column and filters only side effects.
    Prints the number of rows before and after filtering.
    """
    if 'text' not in df.columns:
        raise ValueError("DataFrame must contain 'text' column")

    initial_rows = len(df)

    X = df["text"].astype(str)
    X_vec = _vectorizer.transform(X)

    df["sideeffect_pred"] = _model.predict(X_vec)
    # 1 = sideeffect, 0 = no_sideeffect

    filtered_df = df[df["sideeffect_pred"] == 1].reset_index(drop=True)
    final_rows = len(filtered_df)

    print(f"Initial CSV had {initial_rows} rows; after filtering side effects, {final_rows} rows remain.")

    return filtered_df

# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    """
    Clean raw text by removing URLs, mentions, special characters,
    and extra whitespace.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()

# -----------------------------
# Symptom extraction
# -----------------------------
def extract_symptoms(text, lexicon=None):
    """
    Extracts symptoms from text based on a given lexicon.
    If no lexicon is provided, uses default known + uncommon symptoms.
    """
    if lexicon is None:
        lexicon = [
            "spike protein", "diabetes", "vascular", "autoimmune", "p53",
            "t cell", "vitamin d", "contaminated", "zinc", "circumin",
            "ivermectin", "cancer", "myocarditis", "hospital", "outpatient",
            "inpatient", "infection", "bacteria", "fungal", "viral", "sepsis",
            "respiratory", "gynaecology", "dermatology", "ophthalmology",
            "otology", "dental", "hypoxia", "doxycycline", "nanosolver",
            "anticoagulation", "aspirin", "stroke", "heart attack",
            "coronary artery disease", "arrhythmia", "autism",
            "neurodegenerative", "alzheimer", "cognitive",
            "vascular dementia", "parkinson", "immune", "pots", "mcas",
            "insomnia", "new onset dyslipidaemia", "hypertension",
            "cardio-metabolic disturbance", "nervous system",
            "mast cell activity in skin", "post-covid-19 vaccine syndrome",
            "seizure disorders", "migraines", "neuropathy",
            "inflammatory bowel disease", "depression", "anxiety disorders",
            "chronic fatigue syndrome", "lyme disease", "fibromyalgia",
            "arthritis", "chronic obstructive pulmonary disease", "copd",
            "asthma", "chronic kidney disease", "ckd", "chronic heart failure",
            "chf", "bleeding disorders", "atherosclerosis", "vasculopathies",
            "endothelialitis", "thrombosis", "coagulopathy", "long covid",
            "thrombocytopenia", "low platelet", "internal bleeding",
            "lymphopenia", "neutropenia", "suppressed immune", "immune dysfunction",
            "muscle pain", "joint pain", "vomiting", "fever", "autoimmunity",
            "sleep apnea", "guillian barre syndrome", "adem", "cvst",
            "spike amyloids hamper fibrinolysis", "sticky blood", "neuropsychiatric",
            "mrna", "psychosis", "dementia", "schizophrenia", "suicidal",
            "homicidal", "brain clot", "violent behavior", "cognitive decline",
            "delusion", "takotsubo cardiomyopathy", "lipid nanoparticle toxicity",
            "allergenic", "cytotoxic", "pneumonia", "endocrine",
            "immune microclot", "vascular dysfunction", "teamclot",
            "organ impairment", "endothelian diagnostic", "thromboembolic events",
            "inflammatory cytokine increase", "allergic reactions", "igg increase",
            "iga increase"
        ]

    if not isinstance(text, str):
        return []
    return [s for s in lexicon if s in text.lower()]

# -----------------------------
# Slang normalization
# -----------------------------
slang_lexicon = {
    "feel like shit": ["fatigue", "malaise"],
    "exhausted": ["fatigue"],
    "shield against the storm": ["immune response", "general malaise"],
    "i have been run over by a truck": ["muscle pain", "joint pain", "fatigue"],
    "tired": ["fatigue"],
    "knackered": ["fatigue"],
    "wiped out": ["fatigue"],
    "brain fog": ["cognitive"],
    "sleepy all day": ["insomnia", "fatigue"],
    "my head is pounding": ["headache"],
    "can't sleep": ["insomnia"],
    "heart racing": ["arrhythmia"],
    "out of breath": ["respiratory distress", "fatigue"]
}

def normalize_slang(text, slang_lexicon=slang_lexicon):
    """
    Normalize slang terms based on a provided slang lexicon.
    Returns a list of standard symptoms that correspond to slang.
    """
    if not isinstance(text, str):
        return []

    text_lower = text.lower()
    matched_symptoms = []

    for slang, symptoms in slang_lexicon.items():
        if slang in text_lower:
            matched_symptoms.extend(symptoms)

    return list(set(matched_symptoms))

# -----------------------------
# Extract known/uncommon symptoms
# -----------------------------
def extract_known_symptoms(text):
    known_symptoms_keywords = [
        "fever", "fatigue", "headache", "muscle pain", "joint pain",
        "vomiting", "insomnia", "cognitive", "anxiety disorders",
        "depression", "respiratory", "asthma", "chronic fatigue syndrome",
        "migraine", "neuropathy", "sleep apnea"
    ]
    if not isinstance(text, str):
        return []
    return [s for s in known_symptoms_keywords if s in text.lower()]

def extract_uncommon_symptoms(text):
    uncommon_symptoms_keywords = [
        "myocarditis", "stroke", "heart attack", "coronary artery disease",
        "arrhythmia", "thrombosis", "coagulopathy", "thrombocytopenia",
        "low platelet", "internal bleeding", "lymphopenia", "neutropenia",
        "guillian barre syndrome", "adem", "cvst", "takotsubo cardiomyopathy",
        "lipid nanoparticle toxicity", "brain clot", "psychosis", "schizophrenia",
        "suicidal", "homicidal", "autoimmunity", "vascular dysfunction",
        "immune dysfunction", "organ impairment", "spike amyloids hamper fibrinolysis",
        "sticky blood", "neuropsychiatric", "post-covid-19 vaccine syndrome", "long covid"
    ]
    if not isinstance(text, str):
        return []
    return [s for s in uncommon_symptoms_keywords if s in text.lower()]

# -----------------------------
# Load and process CSV
# -----------------------------
def load_and_process_data(input_file_path, text_column="text"):
    df = pd.read_csv(input_file_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in input file.")

    df["cleaned_text"] = df[text_column].astype(str).apply(clean_text)
    df["extracted_symptoms"] = df["cleaned_text"].apply(extract_known_symptoms)
    df["extracted_uncommon_symptoms"] = df["cleaned_text"].apply(extract_uncommon_symptoms)
    df["known_symptoms_flag"] = df["extracted_symptoms"].apply(lambda x: int(len(x) > 0))
    df["uncommon_symptoms_flag"] = df["extracted_uncommon_symptoms"].apply(lambda x: int(len(x) > 0))

    alert_keywords = ["urgent", "emergency", "severe", "critical", "immediate"]
    df["alert_keywords_flag"] = df["cleaned_text"].apply(lambda x: int(any(k in x for k in alert_keywords)))

    return df

# -----------------------------
# Main pipeline
# -----------------------------
def main(csv_path, text_column="text"):
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(f"CSV must contain a '{text_column}' column")

    # Step 1: Filter side effects
    df = classify_sideeffects(df)

    # Step 2: Clean text
    df["cleaned_text"] = df[text_column].astype(str).apply(clean_text)

    # Step 3: Extract symptoms (full default lexicon)
    df["extracted_symptoms"] = df["cleaned_text"].apply(lambda x: extract_symptoms(x))
    df["extracted_uncommon_symptoms"] = df["cleaned_text"].apply(lambda x: extract_symptoms(x))

    df["known_symptoms_flag"] = df["extracted_symptoms"].apply(lambda x: int(len(x) > 0))
    df["uncommon_symptoms_flag"] = df["extracted_uncommon_symptoms"].apply(lambda x: int(len(x) > 0))

    # Step 4: Detect alert keywords
    alert_keywords = ["urgent", "emergency", "severe", "critical", "immediate"]
    df["alert_keywords_flag"] = df["cleaned_text"].apply(lambda x: int(any(k in x for k in alert_keywords)))

    return df

# -----------------------------
# Onset time processing
# -----------------------------
def extract_onset_time(text):
    if not isinstance(text, str):
        return []
    replacements = {
        "a": 1, "few": 2, "couple": 2, "several": 3, "many": 5,
        "dozen": 12, "half": 0.5, "long": 8, "short": 1, "some": 3,
        "next": 24, "last": 24, "immediate": 0, "soon": 1, "this": 1,
        "after": 1, "before": 1, "morning": 6, "afternoon": 6,
        "evening": 6, "night": 8, "week": 168, "month": 730
    }
    time_patterns = [
        r"(\d+|a|few|couple|several|many|dozen|half)\s*(hours?|days?)\s*(post-dose|after\s*vaccination|after\s*shot|after\s*injection|post-vaccine|post\s*jab)",
        r"(\d+|a|few|couple|several|many|dozen|half)\s*(hour|day)\s*(after|since|post|following)"
    ]
    onset_times = []
    for pattern in time_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            if value in replacements:
                onset_times.append(replacements[value])
            else:
                try:
                    onset_times.append(int(value))
                except ValueError:
                    continue
    return onset_times

def calculate_duration(timestamp, onset_times):
    symptom_duration = []
    if timestamp is pd.NaT or not isinstance(onset_times, list):
        return symptom_duration
    for onset in onset_times:
        if isinstance(onset, (int, float)):
            symptom_duration.append(timestamp + pd.Timedelta(hours=onset))
    return symptom_duration

def process_onset_data(df):
    """
    Prepares the DataFrame for onset/duration plotting.
    Adds 'timestamp', 'onset_time', and 'symptom_duration' columns.
    """
    df = df.copy()
    if 'Date' not in df.columns:
        raise ValueError("CSV must contain a 'Date' column")
    if 'text' not in df.columns:
        raise ValueError("DataFrame must contain 'text' column")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['timestamp'] = df['Date']
    df['onset_time'] = df['text'].apply(extract_onset_time)
    df['symptom_duration'] = df.apply(lambda row: calculate_duration(row['timestamp'], row['onset_time']), axis=1)
    return df
