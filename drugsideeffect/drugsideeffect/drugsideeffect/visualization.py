# Visualization.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from textblob import TextBlob

from drugsideeffect.processing import (
    extract_symptoms,
    extract_known_symptoms,
    extract_uncommon_symptoms,
    normalize_slang
)


# Function for Day of the Week Distribution
def plot_day_of_week_distribution(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.day_name()
    day_of_week_counts = df['DayOfWeek'].value_counts()

    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week_counts = day_of_week_counts.reindex(weekday_order, fill_value=0)

    plt.figure(figsize=(10, 6))
    day_of_week_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Occurrences of Each Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Function for Count of Data per Month
def plot_data_count_per_month(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    df['month'] = df['Date'].dt.month  # Extract month if not already present
    df = df.sort_values(by='Date')

    # Ensure all months are represented in the data
    all_months = pd.Series(range(1, 13))
    monthly_data = df.groupby('month').size().reset_index(name='count')
    monthly_data = all_months.to_frame(name='month').merge(monthly_data, on='month', how='left')

    plt.figure(figsize=(12, 6))
    sns.barplot(x=monthly_data['month'], y=monthly_data['count'], palette='Blues')
    plt.title("Count of Data per Month", fontsize=18)
    plt.xlabel("Month", fontsize=19)
    plt.ylabel("Count of Data", fontsize=19)
    plt.xticks(rotation=45, fontsize=19)
    plt.yticks(fontsize=14)
    plt.show()

# Function to Perform Sentiment Analysis and Plot Distribution
def plot_sentiment_distribution(df):
    def get_sentiment(text):
        if not isinstance(text, str):
            text = ""  # Replace non-string with empty string or some default text

        # Perform sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        # Assign sentiment categories
        if polarity > 0:
            return 'positive'
        elif polarity < 0:
            return 'negative'
        else:
            return 'neutral'

    df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values.astype(int), palette='Set2')
    plt.title('Sentiment Distribution', fontsize=16)
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

from drugsideeffect.processing import extract_known_symptoms


# -----------------------------
# Known symptoms plot
# -----------------------------
def plot_known_symptoms(df):
    """
    Plot a bar chart of known symptoms including lexicons:
    - Known symptoms
    - Uncommon symptoms
    - Slang-based symptoms
    """
    import matplotlib.pyplot as plt

    # Ensure cleaned_text exists
    if "cleaned_text" not in df.columns:
        df["cleaned_text"] = df["text"].astype(str).apply(lambda x: x.lower())

    # Helper to extract all symptoms from all lexicons
    def extract_all_symptoms(text):
        from sideeffect.processing import (
            extract_known_symptoms,
            extract_uncommon_symptoms,
            normalize_slang,
            extract_symptoms
        )

        known = extract_known_symptoms(text)
        uncommon = extract_uncommon_symptoms(text)
        slang = normalize_slang(text, slang_lexicon)
        default = extract_symptoms(text)  # Uses default lexicon

        # Combine all without duplicates
        return list(set(known + uncommon + slang + default))

    # Apply extraction
    df["all_known_symptoms"] = df["cleaned_text"].apply(extract_all_symptoms)

    # Count occurrences
    counts = {}
    for lst in df["all_known_symptoms"]:
        for symptom in lst:
            counts[symptom] = counts.get(symptom, 0) + 1

    # Plot bar chart
    if counts:
        plt.figure(figsize=(10, 6))
        plt.bar(counts.keys(), counts.values())
        plt.title("Distribution of All Known & Slang Symptoms")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print("No known symptoms found to plot.")



# -----------------------------
# Uncommon symptoms plot
# -----------------------------
def plot_uncommon_side_effects_pie_chart(df):
    import matplotlib.pyplot as plt

    if "cleaned_text" not in df.columns:
        df["cleaned_text"] = df["text"].astype(str).str.lower()

    df["extracted_uncommon_plot"] = df["cleaned_text"].apply(extract_uncommon_symptoms)

    counts = {}
    for lst in df["extracted_uncommon_plot"]:
        for symptom in lst:
            counts[symptom] = counts.get(symptom, 0) + 1

    if not counts:
        print("No uncommon symptoms found to plot.")
        return

    plt.figure(figsize=(7, 7))
    plt.pie(counts.values(), labels=counts.keys(),
            autopct="%1.1f%%", startangle=90)
    plt.title("Uncommon Side Effects Distribution")
    plt.show()




# -----------------------------
# Pie chart for overall symptom distribution
# -----------------------------
def plot_symptom_distribution(df):
    import matplotlib.pyplot as plt

    if "cleaned_text" not in df.columns:
        df["cleaned_text"] = df["text"].astype(str).apply(lambda x: x.lower())

    # Use safe wrappers from processing.py
    df["extracted_symptoms_plot"] = df["cleaned_text"].apply(extract_known_symptoms)
    df["extracted_uncommon_plot"] = df["cleaned_text"].apply(extract_uncommon_symptoms)

    common_symptoms_count = df["extracted_symptoms_plot"].apply(len).gt(0).sum()
    uncommon_symptoms_count = df["extracted_uncommon_plot"].apply(len).gt(0).sum()
    alert_keywords_count = df.get('alert_keywords_flag', pd.Series([0]*len(df))).sum()

    labels = ["Common Symptoms", "Uncommon Symptoms", "Alert Keywords"]

    plt.figure(figsize=(7, 7))
    plt.pie(
        [common_symptoms_count, uncommon_symptoms_count, alert_keywords_count],
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=["#ff9999", "#66b3ff", "#ffcc00"]
    )
    plt.title("Distribution of Symptoms and Alerts")
    plt.show()









# Function to plot the count of data per month
def plot_data_count_per_month(df):
    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    
    # Extract month if not already present
    df['month'] = df['Date'].dt.month
    df = df.sort_values(by='Date')  # Sort data by Date to apply rolling average correctly

    # Set larger font sizes for the plots
    sns.set_context("notebook", font_scale=1.5)  # Increase the font size of labels and titles

    # Ensure all months are represented (even months with no records)
    all_months = pd.Series(range(1, 13))
    monthly_data = df.groupby('month').size().reset_index(name='count')
    monthly_data = all_months.to_frame(name='month').merge(monthly_data, on='month', how='left')

    # Plot the data count per month
    plt.figure(figsize=(12, 6))
    sns.barplot(x=monthly_data['month'], y=monthly_data['count'], palette='Blues')
    plt.title("Count of Data per Month", fontsize=18)
    plt.xlabel("Month", fontsize=19)
    plt.ylabel("Count of Data", fontsize=19)
    plt.xticks(rotation=45, fontsize=19)
    plt.yticks(fontsize=14)
    plt.show()









from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to get sentiment (positive, neutral, negative) from text
def get_sentiment(text):
    # Ensure the input is a string
    if not isinstance(text, str):
        text = ""  # Replace non-string with empty string or some default text

    # Perform sentiment analysis
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Sentiment polarity: ranges from -1 (negative) to 1 (positive)

    # Assign sentiment categories based on polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# New function to plot the sentiment distribution of tweets
def plot_sentiment_distribution(df):
    # Apply sentiment analysis to each tweet in 'cleaned_text' column
    df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

    # Check sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()

    # Create a bar plot to show the distribution of sentiment categories
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values.astype(int), palette='Set2')

    # Customize plot labels and title
    plt.title('Sentiment Distribution of Tweets', fontsize=16)
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot
    plt.show()






import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Visualization: Tweet activity by hour of the day
def plot_tweet_activity_by_hour(df):
    # Extract the hour from the 'Date' column
    df['hour'] = df['Date'].dt.hour

    # Visualizing tweet activity by hour of the day
    plt.figure(figsize=(12, 6))
    sns.countplot(x='hour', data=df, palette='coolwarm')
    plt.title('Tweet Activity by Hour (Vaccine Side Effects Mentioned)', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel('Number of Tweets', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()









import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from textblob import TextBlob

# Function to extract timestamp mentions and track onset (from the Text0 column)
def extract_onset_time(text):
    if not isinstance(text, str):  # Ensure text is a string
        return []  # Return empty list for non-string values (including NaN/None)

    # Define replacement for non-numeric time values (e.g., "a", "few", "couple")
    replacements = {
        "a": 1,           # "a" becomes 1 hour
        "few": 2,         # "few" becomes 2 hours
        "couple": 2,      # "couple" becomes 2 hours
        "several": 3,     # "several" becomes 3 hours
        "many": 5,        # "many" becomes 5 hours
        "dozen": 12,      # "dozen" becomes 12 hours
        "half": 0.5,      # "half" becomes 0.5 hours
        "long": 8,        # "long" becomes 8 hours
        "short": 1,       # "short" becomes 1 hour
        "some": 3,        # "some" becomes 3 hours
        "next": 24,       # "next" becomes 24 hours (1 day)
        "last": 24,       # "last" becomes 24 hours (1 day)
        "immediate": 0,   # "immediate" becomes 0 hours (instantaneous)
        "soon": 1,        # "soon" becomes 1 hour
        "this": 1,        # "this" becomes 1 hour
        "after": 1,       # "after" becomes 1 hour
        "before": 1,      # "before" becomes 1 hour
        "morning": 6,     # "morning" becomes 6 hours
        "afternoon": 6,   # "afternoon" becomes 6 hours
        "evening": 6,     # "evening" becomes 6 hours
        "night": 8,       # "night" becomes 8 hours
        "week": 168,      # "week" becomes 168 hours (7 days)
        "month": 730,     # "month" becomes 730 hours (30 days)
    }

    time_patterns = [
        r"(\d+)\s*(hours?|days?)\s*(post-dose|after\s*vaccination|after\s*shot|after\s*injection|post-vaccine|post\s*jab)",
        r"(\d+)\s*(hour|day)\s*(after|since|post|following)",
        # Add more patterns as needed
    ]

    onset_times = []
    for pattern in time_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).lower()  # Match group as a lowercase string
            if value in replacements:
                onset_times.append(replacements[value])
            else:
                try:
                    onset_times.append(int(value))  # Try converting numeric values
                except ValueError:
                    continue  # In case of unexpected non-numeric values

    return onset_times

# Function to analyze symptom duration based on timestamps
def calculate_duration(timestamp, onset_times):
    symptom_duration = []
    for onset in onset_times:
        if isinstance(onset, int):  # Ensure onset is numeric (hours or days)
            duration = timestamp + pd.Timedelta(hours=onset)
            symptom_duration.append(duration)
    return symptom_duration

# Visualization: Plot results based on symptom onset times and symptom durations
def plot_results(df):
    # Step 1: Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M', errors='coerce')

    # Step 2: Extract year, month, and day from 'Date'
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day

    # Step 3: Create the timestamp column from the year, month, and day columns
    df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day']])  # Combine year, month, day

    # Step 4: Apply the onset extraction function to the 'Text0' column (assumed to contain symptom descriptions)
    df['onset_time'] = df['text'].apply(extract_onset_time)

    # Step 5: Apply duration calculation to determine symptom duration
    df['symptom_duration'] = df.apply(lambda row: calculate_duration(row['timestamp'], row['onset_time']), axis=1)

    # Plot distribution of onset times
    onset_times_flat = [item for sublist in df['onset_time'] for item in sublist]  # Flatten the list of onset times
    plt.figure(figsize=(10, 6))
    sns.histplot(onset_times_flat, kde=True, color='skyblue', bins=10)
    plt.title("Distribution of Symptom Onset Times (in Hours)")
    plt.xlabel("Onset Time (Hours)")
    plt.ylabel("Frequency")
    plt.show()

    # Plot distribution of symptom durations (in hours)
    symptom_durations_flat = [item for sublist in df['symptom_duration'] for item in sublist]  # Flatten the list of durations
    plt.figure(figsize=(10, 6))
    sns.histplot(symptom_durations_flat, kde=True, color='salmon', bins=10)
    plt.title("Distribution of Symptom Durations (in Hours)")
    plt.xlabel("Duration (Hours)")
    plt.ylabel("Frequency")
    plt.show()

    # Plot the count of symptoms reported over time (based on the timestamp)
    plt.figure(figsize=(10, 6))
    df['date'] = df['timestamp'].dt.date  # Extract just the date from timestamp
    df.groupby('date').size().plot(kind='line', marker='o', color='green')
    plt.title("Number of Symptom Reports Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Reports")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()







import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict
from textblob import TextBlob

# Assuming the uncommon vaccine side effects lexicon is predefined as:
uncommon_vaccine_side_effects_lexicon = [
        "myocarditis", "stroke", "heart attack", "coronary artery disease",
        "arrhythmia", "thrombosis", "coagulopathy", "thrombocytopenia",
        "low platelet", "internal bleeding", "lymphopenia", "neutropenia",
        "guillian barre syndrome", "adem", "cvst", "takotsubo cardiomyopathy",
        "lipid nanoparticle toxicity", "brain clot", "psychosis", "schizophrenia",
        "suicidal", "homicidal", "autoimmunity", "vascular dysfunction",
        "immune dysfunction", "organ impairment", "spike amyloids hamper fibrinolysis",
        "sticky blood", "neuropsychiatric", "post-covid-19 vaccine syndrome", "long covid"
    ]



# Function to extract timestamp mentions and track onset (from the Text0 column)
def extract_onset_time(text):
    if not isinstance(text, str):  # Ensure text is a string
        return []  # Return empty list for non-string values (including NaN/None)

    # Define replacement for non-numeric time values (e.g., "a", "few", "couple")
    replacements = {
        "a": 1,           # "a" becomes 1 hour
        "few": 2,         # "few" becomes 2 hours
        "couple": 2,      # "couple" becomes 2 hours
        "several": 3,     # "several" becomes 3 hours
        "many": 5,        # "many" becomes 5 hours
        "dozen": 12,      # "dozen" becomes 12 hours
        "half": 0.5,      # "half" becomes 0.5 hours
        "long": 8,        # "long" becomes 8 hours
        "short": 1,       # "short" becomes 1 hour
        "some": 3,        # "some" becomes 3 hours
        "next": 24,       # "next" becomes 24 hours (1 day)
        "last": 24,       # "last" becomes 24 hours (1 day)
        "immediate": 0,   # "immediate" becomes 0 hours (instantaneous)
        "soon": 1,        # "soon" becomes 1 hour
        "this": 1,        # "this" becomes 1 hour
        "after": 1,       # "after" becomes 1 hour
        "before": 1,      # "before" becomes 1 hour
        "morning": 6,     # "morning" becomes 6 hours
        "afternoon": 6,   # "afternoon" becomes 6 hours
        "evening": 6,     # "evening" becomes 6 hours
        "night": 8,       # "night" becomes 8 hours
        "week": 168,      # "week" becomes 168 hours (7 days)
        "month": 730,     # "month" becomes 730 hours (30 days)
    }

    time_patterns = [
        r"(\d+)\s*(hours?|days?)\s*(post-dose|after\s*vaccination|after\s*shot|after\s*injection|post-vaccine|post\s*jab)",
        r"(\d+)\s*(hour|day)\s*(after|since|post|following)",
        # Add more patterns as needed
    ]

    onset_times = []
    for pattern in time_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).lower()  # Match group as a lowercase string
            if value in replacements:
                onset_times.append(replacements[value])
            else:
                try:
                    onset_times.append(int(value))  # Try converting numeric values
                except ValueError:
                    continue  # In case of unexpected non-numeric values

    return onset_times

# Function to analyze symptom duration based on timestamps
def calculate_duration(timestamp, onset_times):
    symptom_duration = []
    for onset in onset_times:
        if isinstance(onset, int):  # Ensure onset is numeric (hours or days)
            duration = timestamp + pd.Timedelta(hours=onset)
            symptom_duration.append(duration)
    return symptom_duration

# Visualization: Plot results based on symptom onset times and symptom durations
def plot_results(df):
    # Step 1: Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M', errors='coerce')

    # Step 2: Extract year, month, and day from 'Date'
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day

    # Step 3: Create the timestamp column from the year, month, and day columns
    df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day']])  # Combine year, month, day

    # Step 4: Apply the onset extraction function to the 'Text0' column (assumed to contain symptom descriptions)
    df['onset_time'] = df['text'].apply(extract_onset_time)

    # Step 5: Apply duration calculation to determine symptom duration
    df['symptom_duration'] = df.apply(lambda row: calculate_duration(row['timestamp'], row['onset_time']), axis=1)

    # Plot distribution of onset times
    onset_times_flat = [item for sublist in df['onset_time'] for item in sublist]  # Flatten the list of onset times
    plt.figure(figsize=(10, 6))
    sns.histplot(onset_times_flat, kde=True, color='skyblue', bins=10)
    plt.title("Distribution of Symptom Onset Times (in Hours)")
    plt.xlabel("Onset Time (Hours)")
    plt.ylabel("Frequency")
    plt.show()

    # Plot distribution of symptom durations (in hours)
    symptom_durations_flat = [item for sublist in df['symptom_duration'] for item in sublist]  # Flatten the list of durations
    plt.figure(figsize=(10, 6))
    sns.histplot(symptom_durations_flat, kde=True, color='salmon', bins=10)
    plt.title("Distribution of Symptom Durations (in Hours)")
    plt.xlabel("Duration (Hours)")
    plt.ylabel("Frequency")
    plt.show()

    # Plot the count of symptoms reported over time (based on the timestamp)
    plt.figure(figsize=(10, 6))
    df['date'] = df['timestamp'].dt.date  # Extract just the date from timestamp
    df.groupby('date').size().plot(kind='line', marker='o', color='green')
    plt.title("Number of Symptom Reports Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Reports")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Symptom Extraction and Day of Week Analysis
        

    # Check if there's a 'Date' column and create a 'day_of_week' column from it
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_week'] = df['day_of_week'].map({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
                                                   3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'})
    else:
        df['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_week'] = df['day_of_week'].map({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
                                                   3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'})

    # Flatten the 'extracted_symptoms' list column
    df_flat = df.explode('extracted_symptoms')

    # Group by 'day_of_week' and 'extracted_symptoms', and count occurrences
    symptom_by_time = df_flat.groupby(['day_of_week', 'extracted_symptoms']).size().unstack().fillna(0)

    # Define the correct order of the days of the week
    days_of_week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Reorder the columns and index to make sure the days of the week are in the correct order
    symptom_by_time = symptom_by_time.reindex(days_of_week_order)

    # Find the top 10 symptoms based on the total count across all days
    top_10_symptoms = symptom_by_time.sum(axis=0).nlargest(10).index

    # Filter the DataFrame to only include the top 10 symptoms
    symptom_by_time_top_10 = symptom_by_time[top_10_symptoms]

    # Plot the stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 7))  # Reduced figure size
    symptom_by_time_top_10.plot(kind='bar', stacked=True, ax=ax)

    # Title and labels
    plt.title('Symptom Occurrence Over Days of the Week (Top 10 Symptoms)', fontsize=14)
    plt.ylabel('Number of Occurrences', fontsize=12)

    # Rotate x-ticks for better readability
    plt.xticks(rotation=45)

    # Place the legend outside of the plot to avoid cluttering the image
    plt.legend(title='Symptoms', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to make space for the legend
    plt.tight_layout()
    plt.show()

    # Calculate the total occurrences for each day (row sums)
    total_per_day = symptom_by_time_top_10.sum(axis=1)

    # Calculate the proportion of each symptom for each day
    proportions_by_day = symptom_by_time_top_10.div(total_per_day, axis=0)

    # Display the proportions in tabular form
    print("Proportions of Top 10 Symptoms by Day of the Week:")
    print(proportions_by_day)

    # Optionally, display the proportions in a table in a subplot below the chart
    fig, ax = plt.subplots(figsize=(12, 7))  # Reduced figure size to match previous chart
    ax.axis('tight')  # Hide the axis
    ax.axis('off')  # Hide the axis
    ax.table(cellText=proportions_by_day.round(2).values,  # Show proportions rounded to 2 decimal places
             colLabels=proportions_by_day.columns,
             rowLabels=proportions_by_day.index,
             loc='center',
             cellLoc='center',
             colColours=['#f1f1f1']*len(proportions_by_day.columns))  # Optional color for table header

    # Show the table
    plt.tight_layout()
    plt.show()

    # Plot the heatmap of proportions
    plt.figure(figsize=(10, 7))  # Slightly smaller figure size for heatmap
    sns.heatmap(proportions_by_day, annot=True, cmap='YlGnBu', fmt=".2f", cbar_kws={'label': 'Proportion'})
    plt.title('Proportion of Top 10 Symptoms Across Days of the Week', fontsize=14)
    plt.xlabel('Symptoms', fontsize=12)
    plt.ylabel('Days of the Week', fontsize=12)
    plt.tight_layout()
    plt.show()












import pandas as pd
import re
import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK resources
nltk.download('punkt')  # Tokenization resource
nltk.download('wordnet')  # Lemmatization resource
nltk.download('words')  # English dictionary
nltk.download('punkt_tab')  # Additional tokenizer resource (this might be required)

# Function to clean text (with error handling for non-string values)
def clean_text(text):
    if not isinstance(text, str):  # Check if the value is a string
        return ""  # Return empty string for non-string values
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and digits
    text = text.lower()  # Convert text to lowercase
    return text

# Function to lemmatize the text
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words_in_text = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words_in_text]
    return ' '.join(lemmatized_words)

# Function to check if a word is in the English dictionary
def is_english_word(word):
    return word in english_words_set

# Get the set of English words from NLTK's word corpus
english_words_set = set(words.words())

# Function to plot the proportion of English words
def plot_proportion_of_english_words(df):
    # Clean and lemmatize the Text0 column
    df['cleaned_text'] = df['text'].apply(lambda x: clean_text(x))

    # Check if cleaned_text is valid and apply lemmatization only to valid rows
    df['lemmatized_text'] = df['cleaned_text'].apply(lambda x: lemmatize_text(x) if isinstance(x, str) else '')

    # Tokenize the lemmatized text and check if each word is in the dictionary
    df['word_list'] = df['lemmatized_text'].apply(lambda x: word_tokenize(x) if isinstance(x, str) else [])

    # Flatten the word list for all tweets into a single list
    all_words = [word for sublist in df['word_list'] for word in sublist]

    # Check which words are valid English words
    valid_words = [word for word in all_words if is_english_word(word)]

    # Calculate the proportion of valid words
    valid_word_proportion = len(valid_words) / len(all_words) if len(all_words) > 0 else 0  # Avoid division by zero

    # Visualize the result using a pie chart
    labels = ['English Words', 'Non English Words']  # Adjusted label names to avoid "valid"

    sizes = [len(valid_words), len(all_words) - len(valid_words)]

    plt.figure(figsize=(8, 8))

    # Increase font size of labels and percentages in the pie chart
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=90,
            textprops={'fontsize': 15})  # Change textprops for label font size

    # Title with larger font size
    plt.title(f"Proportion of English Words\n({valid_word_proportion*100:.2f}% English)", fontsize=18, fontweight='bold')

    # Show the plot
    plt.show()

    # Print the proportion of English words
    print(f"Proportion of English words: {valid_word_proportion*100:.2f}%")














import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Function to classify sentiment as Positive, Negative, or Neutral based on polarity
def classify_sentiment(text):
    # Calculate sentiment polarity using TextBlob
    polarity = TextBlob(text).sentiment.polarity

    # Classify sentiment based on polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Function to plot sentiment distribution
def plot_sentiment_distribution(df):
    # Apply the classify_sentiment function to the 'Text0' column to create a new 'sentiment' column
    df['sentiment'] = df['text'].apply(classify_sentiment)

    # Count the number of tweets in each sentiment category
    sentiment_counts = df['sentiment'].value_counts()

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')

    # Set labels and title
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count of Tweets', fontsize=12)
    plt.title('Tweet Sentiment Distribution', fontsize=14)

    # Display the plot
    plt.tight_layout()
    plt.show()











import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import numpy as np
import warnings

# Define lexicons and patterns inside the script
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

#time_patterns = ['morning', 'afternoon', 'night', 'after', 'onset']
time_patterns = [
        r"(\d+|a|few|couple|several|many|dozen|half)\s*(hours?|days?)\s*(post-dose|after\s*vaccination|after\s*shot|after\s*injection|post-vaccine|post\s*jab)",
        r"(\d+|a|few|couple|several|many|dozen|half)\s*(hour|day)\s*(after|since|post|following)"
    ]
#known_symptoms = ['fever', 'headache', 'dizziness', 'chills', 'fatigue']
known_symptoms = [
        "fever", "fatigue", "headache", "muscle pain", "joint pain",
        "vomiting", "insomnia", "cognitive", "anxiety disorders",
        "depression", "respiratory", "asthma", "chronic fatigue syndrome",
        "migraine", "neuropathy", "sleep apnea"
    ]
#uncommon_vaccine_side_effects_lexicon = ['rash', 'fever', 'headache', 'nausea', 'tiredness']
uncommon_vaccine_side_effects_lexicon = [
        "myocarditis", "stroke", "heart attack", "coronary artery disease",
        "arrhythmia", "thrombosis", "coagulopathy", "thrombocytopenia",
        "low platelet", "internal bleeding", "lymphopenia", "neutropenia",
        "guillian barre syndrome", "adem", "cvst", "takotsubo cardiomyopathy",
        "lipid nanoparticle toxicity", "brain clot", "psychosis", "schizophrenia",
        "suicidal", "homicidal", "autoimmunity", "vascular dysfunction",
        "immune dysfunction", "organ impairment", "spike amyloids hamper fibrinolysis",
        "sticky blood", "neuropsychiatric", "post-covid-19 vaccine syndrome", "long covid"
    ]

# Function to categorize slang terms based on the slang_lexicon
def categorize_slang_time_sideeffects(text):
    matched_slangs = [slang for slang, symptoms in slang_lexicon.items() if any(s in text.lower() for s in symptoms)]
    return matched_slangs

# Function to categorize time-based expressions based on time_patterns lexicon
def categorize_time_patterns(text):
    matched_patterns = [pattern for pattern in time_patterns if re.search(pattern, text, re.IGNORECASE)]
    return matched_patterns

# Function to apply sentiment analysis using TextBlob
def get_sentiment(text):
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0:
            return 'Positive'
        elif polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'
    except Exception as e:
        warnings.warn(f"Error in sentiment analysis: {e}")
        return 'Unknown'

# Combined plot function for both Slang Sentiment Distribution and Time Series Analysis
def plot_slang_and_time_series(df):
    try:
        # Apply the categorization function to identify slang matches and time-based patterns in the 'Text0' column
        df['slang_matches'] = df['text'].apply(lambda x: categorize_slang_time_sideeffects(x))
        df['time_matches'] = df['text'].apply(lambda x: categorize_time_patterns(x))

        # Explode slang and time matches into separate rows
        df_exploded = df.explode('slang_matches').explode('time_matches')

        # Apply sentiment analysis on the text
        df_exploded['sentiment'] = df_exploded['text'].apply(get_sentiment)

        # 1. Frequency of Slang Terms with Sentiment and Time Patterns - Bar Chart
        slang_sentiment_counts = df_exploded.groupby(['slang_matches', 'month', 'sentiment']).size().reset_index(name='count')
        time_sentiment_counts = df_exploded.groupby(['time_matches', 'month', 'sentiment']).size().reset_index(name='count')

        # Limit to top 10 slang terms based on count for better visualization
        top_10_slangs = slang_sentiment_counts.groupby('slang_matches')['count'].sum().nlargest(10).index
        slang_sentiment_counts = slang_sentiment_counts[slang_sentiment_counts['slang_matches'].isin(top_10_slangs)]

        top_10_times = time_sentiment_counts.groupby('time_matches')['count'].sum().nlargest(10).index
        time_sentiment_counts = time_sentiment_counts[time_sentiment_counts['time_matches'].isin(top_10_times)]

        # Modify x-axis labels to prefix the month number and limit the label length to 30 characters
        slang_sentiment_counts['x_label'] = slang_sentiment_counts['month'].astype(str) + ' ' + slang_sentiment_counts['slang_matches']
        slang_sentiment_counts['x_label'] = slang_sentiment_counts['x_label'].apply(lambda x: (x[:30] if len(x) > 30 else x))  # Limit to 30 characters

        time_sentiment_counts['x_label'] = time_sentiment_counts['month'].astype(str) + ' ' + time_sentiment_counts['time_matches']
        time_sentiment_counts['x_label'] = time_sentiment_counts['x_label'].apply(lambda x: (x[:30] if len(x) > 30 else x))  # Limit to 30 characters

        # Plot Frequency of Slang Terms with Sentiment and Month - Stacked Bar Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=slang_sentiment_counts, x='x_label', y='count', hue='sentiment', palette='Set2', ci=None)
        plt.xticks(rotation=90)
        plt.xlabel('Slang Term', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Frequency of Slang Terms with Sentiment and Month', fontsize=16)
        plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, labels=[(item[:30] + '...') for item in plt.gca().get_legend_handles_labels()[1]])
        plt.tight_layout()
        plt.show()

        # Plot Frequency of Time Patterns with Sentiment and Month - Stacked Bar Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=time_sentiment_counts, x='x_label', y='count', hue='sentiment', palette='Set2', ci=None)
        plt.xticks(rotation=90)
        plt.xlabel('Time Pattern', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Frequency of Time Patterns with Sentiment and Month', fontsize=16)
        plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, labels=[(item[:30] + '...') for item in plt.gca().get_legend_handles_labels()[1]])
        plt.tight_layout()
        plt.show()

        # 2. Time Series Analysis of Slang Terms Mentions - Line Plot
        slang_mentions_by_month = df_exploded.groupby(['month', 'slang_matches']).size().reset_index(name='mention_count')

        # Limit to top 10 slang terms for visualization
        top_10_slangs_month = slang_mentions_by_month.groupby('slang_matches')['mention_count'].sum().nlargest(10).index
        slang_mentions_by_month = slang_mentions_by_month[slang_mentions_by_month['slang_matches'].isin(top_10_slangs_month)]

        # Plot Time Series of Slang Mentions
        plt.figure(figsize=(14, 8))
        sns.lineplot(data=slang_mentions_by_month, x='month', y='mention_count', hue='slang_matches', marker='o', palette='Set2', lw=2)
        plt.title('Time Series of Slang Terms Mentions', fontsize=16)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Mention Count', fontsize=12)
        plt.legend(title='Slang Term', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, labels=[(item[:30] + '...') for item in plt.gca().get_legend_handles_labels()[1]])
        plt.tight_layout()
        plt.show()

        # 3. Time Series Analysis of Time Pattern Mentions - Line Plot
        time_mentions_by_month = df_exploded.groupby(['month', 'time_matches']).size().reset_index(name='mention_count')

        # Limit to top 10 time patterns for visualization
        top_10_times_month = time_mentions_by_month.groupby('time_matches')['mention_count'].sum().nlargest(10).index
        time_mentions_by_month = time_mentions_by_month[time_mentions_by_month['time_matches'].isin(top_10_times_month)]

        # Plot Time Series of Time Pattern Mentions
        plt.figure(figsize=(14, 8))
        sns.lineplot(data=time_mentions_by_month, x='month', y='mention_count', hue='time_matches', marker='o', palette='Set2', lw=2)
        plt.title('Time Series of Time Pattern Mentions', fontsize=16)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Mention Count', fontsize=12)
        plt.legend(title='Time Pattern', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, labels=[(item[:30] + '...') for item in plt.gca().get_legend_handles_labels()[1]])
        plt.tight_layout()
        plt.show()

    except Exception as e:
        warnings.warn(f"An error occurred while plotting: {e}")










import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta

# Slang lexicon and time pattern labels (define these variables)
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



#time_patterns = ['morning', 'afternoon', 'night', 'after', 'onset']
time_patterns = [
        r"(\d+|a|few|couple|several|many|dozen|half)\s*(hours?|days?)\s*(post-dose|after\s*vaccination|after\s*shot|after\s*injection|post-vaccine|post\s*jab)",
        r"(\d+|a|few|couple|several|many|dozen|half)\s*(hour|day)\s*(after|since|post|following)"
    ]
time_pattern_labels = {
    'morning': 'Morning',
    'afternoon': 'Afternoon',
    'night': 'Night',
    'after': 'After',
    'onset': 'Onset'
}

# Function to categorize slang lexicon and check for time patterns
def categorize_slang_and_time_patterns(text):
    try:
        # Check for slang matches
        matched_slangs = []
        for slang, patterns in slang_lexicon.items():
            if any(pattern.lower() in text.lower() for pattern in patterns):
                matched_slangs.append(slang)

        # Check for time pattern matches and map them to human-readable labels
        time_pattern_matches = []
        for pattern in time_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Map the pattern to its corresponding label from time_pattern_labels
                label = time_pattern_labels.get(pattern, pattern)  # Default to pattern if no label is found
                time_pattern_matches.append(label)

        return matched_slangs, time_pattern_matches
    except Exception as e:
        print(f"Error in categorize_slang_and_time_patterns: {e}")
        return [], []

# Function to create the plot based on the dataset
def plot_create_slang_time_pattern(dataframe):
    try:
        # Apply the function to categorize slang and detect time patterns in the 'Text0' column
        dataframe[['slang_matches', 'time_pattern_matches']] = dataframe['text'].apply(lambda x: pd.Series(categorize_slang_and_time_patterns(x)))

        # Filter rows with slang matches and avoid exploding
        df_slang = dataframe[dataframe['slang_matches'].apply(lambda x: len(x) > 0)]

        # Exploding the lists into individual rows for slang and time patterns
        df_slang_exploded = df_slang.copy()
        df_slang_exploded = df_slang_exploded.explode('slang_matches')
        df_slang_exploded = df_slang_exploded.explode('time_pattern_matches')

        # Add month information
        df_slang_exploded['month'] = pd.to_datetime(df_slang_exploded['Date'], errors='coerce').dt.month

        # Drop rows with NaT (invalid dates)
        df_slang_exploded = df_slang_exploded.dropna(subset=['month'])

        # Group by month, slang match, and time pattern to count occurrences
        monthly_slang_counts = df_slang_exploded.groupby(['month', 'slang_matches', 'time_pattern_matches']).size().reset_index(name='count')

        # Get the top slang expressions per month (top 10 as an example)
        top_slangs_per_month = monthly_slang_counts.groupby(['month', 'slang_matches'], group_keys=False).apply(
            lambda x: x.nlargest(10, 'count')  # Get top 10 slang expressions per month
        ).reset_index(drop=True)

        # Create a pivot table for the plot
        pivot_table = top_slangs_per_month.pivot_table(index=['month', 'slang_matches'],
                                                       columns='time_pattern_matches',
                                                       values='count',
                                                       aggfunc='sum',
                                                       fill_value=0)

        # Reset the index for easier plotting
        pivot_table.reset_index(inplace=True)

        # Combine 'month' and 'slang_matches' to create meaningful x-axis labels
        pivot_table['month_slang'] = pivot_table['month'].astype(str) + ' ' + pivot_table['slang_matches']

        # Plotting the stacked bar plot
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Dynamically filter out columns that do not represent time patterns (e.g., 'month' and 'slang_matches')
        time_pattern_columns = [col for col in pivot_table.columns if col not in ['month', 'slang_matches', 'month_slang']]

        # Plotting the stacked bar chart using only time pattern columns
        pivot_table.set_index('month_slang')[time_pattern_columns].plot(kind='bar', stacked=True, ax=ax1, width=0.8, color=sns.color_palette("Set3", len(time_pattern_columns)))

        # Add labels, title, and customizations
        ax1.set_xlabel("Month & Slang Expression", fontsize=12)
        ax1.set_ylabel("Count of Mentions (Time Patterns)", fontsize=12)
        ax1.set_title("Top Slang Expressions and Associated Time Patterns", fontsize=16)

        # Truncate legend labels to 40 characters max
        truncated_labels = [label if len(label) <= 40 else label[:40] + '...' for label in time_pattern_columns]

        # The legend will now show the truncated time pattern labels
        ax1.legend(title="Time Pattern", labels=truncated_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust x-ticks to show months and slang expressions
        ax1.set_xticklabels(pivot_table['month_slang'], rotation=45, ha='right', fontsize=8)

        # Tight layout for better spacing
        plt.tight_layout()

        # Show the plot
        plt.show()

        return pivot_table  # Return pivot table to save in CSV after the plot
    except Exception as e:
        print(f"Error in plot_create_slang_time_pattern: {e}")
        return None












import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

def plot_create_side_effects_correlation(processed_data):
    # Define the lexicons and known symptoms inside the function
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


    #uncommon_vaccine_side_effects_lexicon = ['fever', 'dizziness', 'nausea']
    uncommon_vaccine_side_effects_lexicon = [
        "myocarditis", "stroke", "heart attack", "coronary artery disease",
        "arrhythmia", "thrombosis", "coagulopathy", "thrombocytopenia",
        "low platelet", "internal bleeding", "lymphopenia", "neutropenia",
        "guillian barre syndrome", "adem", "cvst", "takotsubo cardiomyopathy",
        "lipid nanoparticle toxicity", "brain clot", "psychosis", "schizophrenia",
        "suicidal", "homicidal", "autoimmunity", "vascular dysfunction",
        "immune dysfunction", "organ impairment", "spike amyloids hamper fibrinolysis",
        "sticky blood", "neuropsychiatric", "post-covid-19 vaccine syndrome", "long covid"
    ]

    #known_symptoms = ['rash', 'headache', 'nausea', 'dizziness']
    known_symptoms = [
        "fever", "fatigue", "headache", "muscle pain", "joint pain",
        "vomiting", "insomnia", "cognitive", "anxiety disorders",
        "depression", "respiratory", "asthma", "chronic fatigue syndrome",
        "migraine", "neuropathy", "sleep apnea"
    ]

    # Function to clean the text data
    def clean_text(text):
        if not isinstance(text, str):  # Handle non-string types (e.g., NaN or float)
            return ""  # Convert non-string to empty string or handle it appropriately

        # Clean the text
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#\w+', '', text)  # Remove hashtags
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetical characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

        return text

    # Function to detect side effects based on slang lexicon
    def detect_side_effects(text, slang_lexicon, uncommon_lexicon):
        if not isinstance(text, str):  # Handle non-string inputs (e.g., NaN)
            return []  # Return an empty list if it's not a valid string

        detected_side_effects = []

        # First check the slang lexicon for common side effects
        for slang, effects in slang_lexicon.items():
            if slang in text.lower():
                detected_side_effects.extend(effects)

        # Then check the uncommon side effects lexicon
        for effect in uncommon_lexicon:
            if effect in text.lower():
                detected_side_effects.append(effect)

        return list(set(detected_side_effects))  # Remove duplicates and return the list

    # Clean the text and create a new 'cleaned_text' column
    processed_data['cleaned_text'] = processed_data['text'].apply(lambda x: clean_text(x))

    # Apply the side effect detection function to the 'cleaned_text' column
    processed_data['detected_side_effects'] = processed_data['cleaned_text'].apply(lambda x: detect_side_effects(x, slang_lexicon, uncommon_vaccine_side_effects_lexicon))

    # Create binary flags for common and uncommon side effects
    processed_data['has_common_side_effects'] = processed_data['detected_side_effects'].apply(lambda x: any(effect in known_symptoms for effect in x))
    processed_data['has_uncommon_side_effects'] = processed_data['detected_side_effects'].apply(lambda x: any(effect in uncommon_vaccine_side_effects_lexicon for effect in x))

    # Correlate RT_Like with the detection of side effects (common and uncommon)
    common_side_effects_correlation = processed_data.groupby('has_common_side_effects')['RT_Like'].mean()
    uncommon_side_effects_correlation = processed_data.groupby('has_uncommon_side_effects')['RT_Like'].mean()

    # Visualization 1: Correlation between RT_Like and presence of common/uncommon side effects
    plt.figure(figsize=(12, 6))

    # Plot common side effects correlation with RT_Like
    plt.subplot(1, 2, 1)
    sns.barplot(x=common_side_effects_correlation.index, y=common_side_effects_correlation.values, palette="Blues")
    plt.title("RT_Like Correlation with Common Side Effects")
    plt.xlabel("Common Side Effects Flag (0 = No, 1 = Yes)")
    plt.ylabel("Average RT_Like")

    # Plot uncommon side effects correlation with RT_Like
    plt.subplot(1, 2, 2)
    sns.barplot(x=uncommon_side_effects_correlation.index, y=uncommon_side_effects_correlation.values, palette="Oranges")
    plt.title("RT_Like Correlation with Uncommon Side Effects")
    plt.xlabel("Uncommon Side Effects Flag (0 = No, 1 = Yes)")
    plt.ylabel("Average RT_Like")

    plt.tight_layout()
    plt.show()














import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from textblob import TextBlob  # Import TextBlob for sentiment analysis

def plot_create_side_effects_visualizations(processed_data):
    # Define the severity mapping inside the function
    severity_mapping = {
        'headache': 1, 'sore arm': 1, 'dizziness': 1, 'nausea': 1, 'fatigue': 1, 'brain fog': 1,
        'muscle aches': 1, 'chills': 1, 'feeling unwell': 1, 'fever': 2, 'muscle soreness': 2,
        'joint pain': 2, 'shortness of breath': 2, 'extreme fatigue': 3, 'myocarditis': 3, 'stroke': 4, 'death': 5
    }

    # Define lexicons and time patterns inside the function (you may adjust these based on your use case)
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


    #time_patterns = ['today', 'yesterday', 'last week', 'this week', 'in the past']
    time_patterns = [
        r"(\d+|a|few|couple|several|many|dozen|half)\s*(hours?|days?)\s*(post-dose|after\s*vaccination|after\s*shot|after\s*injection|post-vaccine|post\s*jab)",
        r"(\d+|a|few|couple|several|many|dozen|half)\s*(hour|day)\s*(after|since|post|following)"
    ]
    #known_symptoms = ['rash', 'headache', 'nausea', 'dizziness']
    known_symptoms = [
        "fever", "fatigue", "headache", "muscle pain", "joint pain",
        "vomiting", "insomnia", "cognitive", "anxiety disorders",
        "depression", "respiratory", "asthma", "chronic fatigue syndrome",
        "migraine", "neuropathy", "sleep apnea"
    ]


    #uncommon_vaccine_side_effects_lexicon = ['fever', 'dizziness', 'nausea']
    uncommon_vaccine_side_effects_lexicon = [
        "myocarditis", "stroke", "heart attack", "coronary artery disease",
        "arrhythmia", "thrombosis", "coagulopathy", "thrombocytopenia",
        "low platelet", "internal bleeding", "lymphopenia", "neutropenia",
        "guillian barre syndrome", "adem", "cvst", "takotsubo cardiomyopathy",
        "lipid nanoparticle toxicity", "brain clot", "psychosis", "schizophrenia",
        "suicidal", "homicidal", "autoimmunity", "vascular dysfunction",
        "immune dysfunction", "organ impairment", "spike amyloids hamper fibrinolysis",
        "sticky blood", "neuropsychiatric", "post-covid-19 vaccine syndrome", "long covid"
    ]

    # Function to categorize slang, time patterns, and side effects
    def categorize_slang_time_sideeffects(text):
        matched_slangs = [slang for slang, symptoms in slang_lexicon.items() if any(s in text.lower() for s in symptoms)]
        time_pattern_matches = [pattern for pattern in time_patterns if pattern in text.lower()]
        side_effect_matches = [symptom for symptom in known_symptoms if symptom in text.lower()]
        uncommon_effects_matches = [effect for effect in uncommon_vaccine_side_effects_lexicon if effect in text.lower()]
        return matched_slangs, time_pattern_matches, side_effect_matches, uncommon_effects_matches

    # Apply categorization function to the 'Text0' column
    processed_data[['slang_matches', 'time_pattern_matches', 'side_effect_matches', 'uncommon_effects_matches']] = processed_data['text'].apply(lambda x: pd.Series(categorize_slang_time_sideeffects(x)))

    # Explode matches to separate rows
    df_exploded = processed_data.explode('slang_matches').explode('time_pattern_matches').explode('side_effect_matches').explode('uncommon_effects_matches')

    # Sentiment analysis using TextBlob
    def get_sentiment(text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            return 'Positive'
        elif polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'

    # Apply sentiment analysis to the side effects
    df_exploded['sentiment'] = df_exploded['side_effect_matches'].apply(lambda x: get_sentiment(x) if isinstance(x, str) else 'Neutral')

    # 1. Side Effect Frequency with Sentiment - Bar Chart
    side_effect_sentiment_counts = df_exploded.groupby(['side_effect_matches', 'sentiment']).size().reset_index(name='count')

    # Adjust plot size and label sizes
    plt.figure(figsize=(10, 6))
    sns.barplot(data=side_effect_sentiment_counts, x='side_effect_matches', y='count', hue='sentiment', palette='coolwarm')
    plt.xticks(rotation=90, fontsize=10)  # Reduce the size of x-axis labels
    plt.xlabel('Side Effect', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Side Effect Frequency with Sentiment', fontsize=16)
    plt.tight_layout()
    plt.show()

    # 2. Side Effect Severity Heatmap (without numbers)
    df_exploded['severity'] = df_exploded['side_effect_matches'].apply(lambda x: severity_mapping.get(x, np.nan))  # Use severity mapping to assign severity

    # Group by month and side effect, calculating average severity
    severity_data = df_exploded.groupby(['month', 'side_effect_matches'])['severity'].mean().unstack().fillna(np.nan)  # Ensure missing values are NaN

    # Plot heatmap without annotation (numbers)
    plt.figure(figsize=(12, 8))
    sns.heatmap(severity_data, cmap='YlGnBu', annot=False, cbar_kws={'label': 'Severity'}, linewidths=0.5, mask=severity_data.isna())  # Mask NaN values
    plt.title('Side Effect Severity Heatmap Across Months', fontsize=16)
    plt.xlabel('Side Effect', fontsize=12)
    plt.ylabel('Month', fontsize=12)
    plt.tight_layout()
    plt.show()

    # 3. Time Series of Top 10 Side Effect Mentions with Engagement (RT_Like) - Line Plot
    side_effect_mentions_by_month = df_exploded.groupby(['month', 'side_effect_matches']).size().reset_index(name='mention_count')
    engagement_by_month = df_exploded.groupby(['month', 'side_effect_matches'])['RT_Like'].sum().reset_index(name='total_RT_Like')

    # Merge the two datasets on 'month' and 'side_effect_matches'
    side_effect_time_series = pd.merge(side_effect_mentions_by_month, engagement_by_month, on=['month', 'side_effect_matches'])

    # Normalize Mention Counts and Engagements
    total_mentions = side_effect_time_series['mention_count'].sum()
    total_engagement = side_effect_time_series['total_RT_Like'].sum()

    side_effect_time_series['mention_normalized'] = side_effect_time_series['mention_count'] / total_mentions
    side_effect_time_series['engagement_normalized'] = side_effect_time_series['total_RT_Like'] / total_engagement

    # Compute a Weighted Score for Each Side Effect
    w_mention = 0.7  # Weight for mentions
    w_engagement = 0.3  # Weight for engagement

    # Calculate the weighted score
    side_effect_time_series['score'] = (w_mention * side_effect_time_series['mention_normalized']) + \
                                        (w_engagement * side_effect_time_series['engagement_normalized'])

    # Select Top 10 Side Effects Based on the Score
    top_10_side_effects = side_effect_time_series.groupby('side_effect_matches')['score'].max().reset_index()
    top_10_side_effects = top_10_side_effects.nlargest(10, 'score')

    # Filter the main dataset to include only the top 10 side effects
    top_10_side_effects_list = top_10_side_effects['side_effect_matches'].tolist()
    filtered_data = side_effect_time_series[side_effect_time_series['side_effect_matches'].isin(top_10_side_effects_list)]

    # Plot the Time Series with Mention Counts and Engagement
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=filtered_data, x='month', y='mention_count', hue='side_effect_matches', marker='o', palette='Set2', lw=2)

    # Secondary Y-axis for Engagement (RT_Like)
    ax2 = plt.gca().twinx()
    sns.lineplot(data=filtered_data, x='month', y='total_RT_Like', hue='side_effect_matches', marker='x', ax=ax2, linestyle='--', palette='Set2', lw=2)

    ax2.set_ylabel('Total RT_Like', fontsize=12)  # Clarified label to represent engagement
    plt.title('Time Series of Top 10 Side Effect Mentions with Engagement (RT_Like)', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    ax2.set_ylabel('Total RT_Like', fontsize=12)  # Corrected the right axis label

    # Adjusting the legends
    handles, labels = ax2.get_legend_handles_labels()
    side_effect_legend = plt.legend(handles, labels, title='Side Effect', bbox_to_anchor=(1.05, 1), loc='upper left')
    mention_legend = plt.legend(title="Mentions", loc='upper left', bbox_to_anchor=(1.05, 0.8))
    engagement_legend = plt.legend(title="Engagement (RT_Like)", loc='upper left', bbox_to_anchor=(1.05, 0.6))

    plt.tight_layout()
    plt.show()











import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

def plot_visualize(df):
    # Extracting time pattern matches and slang lexicon matches
    def extract_time_pattern_matches(text, patterns):
        matches = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
        return matches

    def extract_slang_matches(text, slang_dict):
        matches = []
        for key, values in slang_dict.items():
            for slang in values:
                if slang in text:
                    matches.append(key)
        return matches

    # Apply the functions to extract patterns and slang
    df['time_pattern_matches'] = df['text'].apply(lambda x: extract_time_pattern_matches(x, time_patterns))
    df['slang_matches'] = df['text'].apply(lambda x: extract_slang_matches(x, slang_lexicon))

    # Top 5 time patterns and slang lexicons by frequency
    time_pattern_counts = df['time_pattern_matches'].explode().value_counts().head(5)
    slang_counts = df['slang_matches'].explode().value_counts().head(5)

    # Example to show tweet volume for the top 5
    top_time_patterns = time_pattern_counts
    top_slangs = slang_counts

    # Prepare data for plotting
    time_pattern_stats_sorted = pd.DataFrame({
        'time_pattern_matches': top_time_patterns.index,
        'tweet_volume': top_time_patterns.values
    })

    slang_lexicon_stats_sorted = pd.DataFrame({
        'slang_matches': top_slangs.index,
        'tweet_volume': top_slangs.values
    })

    # Calculate the average RT_Like for each time pattern and slang lexicon
    avg_rt_likes_time_patterns = [
        df[df['time_pattern_matches'].apply(lambda x: pattern in x)]['RT_Like'].mean()
        for pattern in top_time_patterns.index
    ]

    avg_rt_likes_slangs = [
        df[df['slang_matches'].apply(lambda x: slang in x)]['RT_Like'].mean()
        for slang in top_slangs.index
    ]

    # Combine the average RT_Likes for both time patterns and slang lexicons
    avg_rt_likes = avg_rt_likes_time_patterns + avg_rt_likes_slangs

    # Calculate the sentiment polarity for each tweet and compute the average sentiment for each pattern/lexicon
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Now calculate the average sentiment for the top 5 time patterns and slang lexicons
    avg_sentiment_time_patterns = [
        df[df['time_pattern_matches'].apply(lambda x: pattern in x)].get('sentiment', pd.Series(dtype='float64')).mean()
        for pattern in top_time_patterns.index
    ]
    avg_sentiment_slangs = [
        df[df['slang_matches'].apply(lambda x: slang in x)].get('sentiment', pd.Series(dtype='float64')).mean()
        for slang in top_slangs.index
    ]

    # Combine the sentiment values with the existing data
    avg_sentiments = avg_sentiment_time_patterns + avg_sentiment_slangs

    # Prepare x labels and tweet volumes for plotting
    x_labels = list(top_time_patterns.index)[:5] + list(top_slangs.index)[:5]
    tweet_volumes = list(time_pattern_stats_sorted['tweet_volume'])[:5] + list(slang_lexicon_stats_sorted['tweet_volume'])[:5]

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Create the bar plot for tweet volume with different colors for time patterns and slang lexicons
    sns.barplot(x=x_labels[:5], y=tweet_volumes[:5], color='skyblue', label='Time Patterns', ax=ax1)

    try:
        # Only plot the second bar plot if there are slang lexicons
        if len(x_labels[5:]) > 0 and len(tweet_volumes[5:]) > 0:
            sns.barplot(x=x_labels[5:], y=tweet_volumes[5:], color='orange', label='Slang Lexicons', ax=ax1)
    except ValueError as e:
        print(f"Error while plotting slang lexicons: {e}")

    # Create the second y-axis for sentiment
    ax2 = ax1.twinx()
    ax2.plot(x_labels, avg_sentiments, color='green', label='Average Sentiment', marker='o', linestyle='--')

    # Create a third y-axis for average RT_Likes
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Shift the third axis to the right
    ax3.plot(x_labels, avg_rt_likes, color='red', label='Average RT_Likes', marker='x', linestyle=':')

    # Set x-ticks and labels
    xticks = list(range(len(x_labels)))  # Generate tick positions based on the number of x_labels
    ax1.set_xticks(xticks)  # Set the positions for the x-ticks
    ax1.set_xticklabels([label[:15] for label in x_labels], rotation=90)  # Set the labels with a limit of 15 characters

    # Add titles and labels
    ax1.set_title('Tweet Volume, Average RT_Likes, and Average Sentiment for Top 5 Time Patterns and Slang Lexicons', fontsize=14)
    ax1.set_xlabel('Time Patterns and Slang Lexicons', fontsize=12)
    ax1.set_ylabel('Tweet Volume', fontsize=12)
    ax2.set_ylabel('Average Sentiment', fontsize=12)
    ax3.set_ylabel('Average RT_Likes', fontsize=12)

    # Display the legend
    ax1.legend(title="Categories", loc='upper left')
    ax2.legend(title="Sentiment", loc='upper right')
    ax3.legend(title="Average RT_Likes", loc='lower right')

    # Show the plot
    plt.tight_layout()
    plt.show()
















import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re



# Function to extract alert keywords based on the alert lexicon
def extract_alert_keywords(text, lexicon):
    try:
        if not isinstance(text, str):  # Ensure text is a string
            return []

        extracted = []
        # Check for alert keywords in the lexicon
        for alert in lexicon:
            if re.search(r'\b' + re.escape(alert) + r'\b', text.lower()):
                extracted.append(alert)

        return extracted
    except Exception as e:
        print(f"Error extracting alert keywords: {e}")
        return []

# Define your lexicons
known_symptoms = ['fever', 'headache', 'chills']  # Example list, adjust as needed
uncommon_vaccine_side_effects_lexicon = ['rash', 'fatigue', 'nausea']  # Example list
alert_keywords = ['urgent', 'serious', 'critical']  # Example list

# New function to perform the symptom extraction visualization
def plot_symptom_extraction(df):
    try:
        # Apply symptom extraction to each tweet (assuming 'text' is the tweet text)
        df['extracted_symptoms'] = df['text'].apply(lambda x: extract_symptoms(x, known_symptoms))

        # Create a flag indicating whether any known symptoms were mentioned in the tweet
        df['known_symptoms_flag'] = df['extracted_symptoms'].apply(lambda x: 1 if x else 0)

        # Apply uncommon symptom extraction to each tweet
        df['extracted_uncommon_symptoms'] = df['text'].apply(lambda x: extract_symptoms(x, uncommon_vaccine_side_effects_lexicon))

        # Create a flag indicating whether any uncommon symptoms were mentioned in the tweet
        df['uncommon_symptoms_flag'] = df['extracted_uncommon_symptoms'].apply(lambda x: 1 if x else 0)

        # Apply alert keyword extraction to each tweet
        df['extracted_alert_keywords'] = df['text'].apply(lambda x: extract_alert_keywords(x, alert_keywords))

        # Create a flag indicating whether any alert keywords were mentioned in the tweet
        df['alert_keywords_flag'] = df['extracted_alert_keywords'].apply(lambda x: 1 if x else 0)

        # Visualization 1: Count of each known symptom detected in the dataset
        all_known_symptoms = [symptom for sublist in df['extracted_symptoms'] for symptom in sublist]
        plt.figure(figsize=(10, 6))
        symptom_counts = pd.Series(all_known_symptoms).value_counts()
        top_10_symptoms = symptom_counts.head(10)
        sns.barplot(x=top_10_symptoms.index, y=top_10_symptoms.values, palette="viridis")
        plt.xticks(rotation=45, ha='right')
        plt.title("Top 10 Known Vaccine Side Effects in Text Data")
        plt.xlabel("Symptom")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        # Visualization 2: Bar chart for uncommon vaccine side effects
        all_uncommon_symptoms = [symptom for sublist in df['extracted_uncommon_symptoms'] for symptom in sublist]
        plt.figure(figsize=(10, 6))
        uncommon_symptom_counts = pd.Series(all_uncommon_symptoms).value_counts()
        top_10_uncommon_symptoms = uncommon_symptom_counts.head(10)
        sns.barplot(x=top_10_uncommon_symptoms.index, y=top_10_uncommon_symptoms.values, palette="plasma")
        plt.xticks(rotation=45, ha='right')
        plt.title("Top 10 Uncommon Vaccine Side Effects in Text Data")
        plt.xlabel("Uncommon Symptom")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        # Visualization 3: Pie chart for distribution of texts with Common Symptoms, Uncommon Symptoms, and Alerts
        common_symptoms_count = (df['known_symptoms_flag'] == 1).sum()
        uncommon_symptoms_count = (df['uncommon_symptoms_flag'] == 1).sum()
        alert_keywords_count = (df['alert_keywords_flag'] == 1).sum()

        labels = ["Common Symptoms", "Uncommon Symptoms", "Alert Keywords"]
        plt.figure(figsize=(7, 7))
        plt.pie([common_symptoms_count, uncommon_symptoms_count, alert_keywords_count], labels=labels,
                autopct='%1.1f%%', startangle=90, colors=["#ff9999", "#66b3ff", "#ffcc00"])
        plt.title("Proportion of Tweets with Common Symptoms vs. Uncommon Symptoms vs. Alerts")
        plt.show()

    except Exception as e:
        print(f"Error in symptom extraction visualization: {e}")


        
        

        
        
        
# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_onset_times(df):
    """
    Plots the distribution of symptom onset times and symptom durations.
    Requires df to have 'onset_time', 'symptom_duration', 'timestamp'.
    """
    # Flatten lists for plotting
    onset_times_flat = [item for sublist in df['onset_time'] for item in sublist if sublist]
    symptom_durations_flat = [
        (d - t).total_seconds() / 3600  # Convert to hours
        for row in df.itertuples()
        for d in row.symptom_duration
        for t in [row.timestamp]
    ]

    # Plot: Symptom onset times
    plt.figure(figsize=(10, 6))
    sns.histplot(onset_times_flat, bins=10, kde=True, color='skyblue')
    plt.title("Distribution of Symptom Onset Times (Hours)")
    plt.xlabel("Onset Time (Hours)")
    plt.ylabel("Frequency")
    plt.show()

    # Plot: Symptom durations
    plt.figure(figsize=(10, 6))
    sns.histplot(symptom_durations_flat, bins=10, kde=True, color='salmon')
    plt.title("Distribution of Symptom Durations (Hours)")
    plt.xlabel("Duration (Hours)")
    plt.ylabel("Frequency")
    plt.show()
