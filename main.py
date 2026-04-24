# ============================================================================
# FULLY UPDATED CODE: Diabetic Exercise Response Analysis
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from scipy import stats          # <-- missing import added
import os

# ----------------------------------------------------------------------------
# Step 1: Import and prepare the data
# ----------------------------------------------------------------------------

# After running the Apple Health parser (e.g., markwk's qs_ledger),
# you should have these CSV files in your working directory:
#   - 'Glucose Data.csv'
#   - 'Workout Data.csv'          (if you have workout events)
#   - optionally 'electrocardiograms/' folder with ECG CSVs

# Load CGM data
glucose_data = 'Glucose Data.csv'
df = pd.read_csv(glucose_data)

# Convert 'Reading date' to datetime
df['Timestamp'] = pd.to_datetime(df['Reading date'])

# Filter out non‑physiological values (<5 mg/dL)
df = df[df['Measurement (mg/dL)'] >= 5]

# Optional: convert mg/dL to mmol/L (uncomment if needed)
# df['BloodSugar_mmol_L'] = df['Measurement (mg/dL)'] * 0.0555

# ----------------------------------------------------------------------------
# Step 2: Load workout data (replace with your actual file if different)
# ----------------------------------------------------------------------------
# The Apple Health parser typically creates 'Workout Data.csv'
workout_data = 'Workout Data.csv'
workout_df = pd.read_csv(workout_data)

# Convert start/end dates to datetime
workout_df['startDate'] = pd.to_datetime(workout_df['startDate'])
workout_df['endDate']   = pd.to_datetime(workout_df['endDate'])

# Optional: filter workout types (e.g., only running, cycling)
# workout_df_filtered = workout_df[workout_df['workoutActivityType'].isin(['HKWorkoutActivityTypeRunning'])]
workout_df_filtered = workout_df.copy()   # use all workouts for now

# ----------------------------------------------------------------------------
# Step 3: Basic visualisations (unchanged from original)
# ----------------------------------------------------------------------------
# Blood sugar over time
plt.figure(figsize=(12, 6))
plt.plot(df['Timestamp'], df['Measurement (mg/dL)'], marker='o', linestyle='-', color='blue', markersize=2)
plt.title('Blood Sugar Levels Over Time')
plt.xlabel('Date')
plt.ylabel('Blood Sugar (mg/dL)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Weekly averages
df['Week'] = df['Timestamp'].dt.isocalendar().week
weekly_avg = df.groupby('Week')['Measurement (mg/dL)'].mean()
plt.figure(figsize=(12, 6))
sns.barplot(x=weekly_avg.index, y=weekly_avg.values, color='green')
plt.title('Average Blood Sugar per Week')
plt.xlabel('Week of the Year')
plt.ylabel('Average Blood Sugar (mg/dL)')
plt.show()

# Day of week averages
df['DayOfWeek'] = df['Timestamp'].dt.day_name()
dayofweek_avg = df.groupby('DayOfWeek')['Measurement (mg/dL)'].mean()
plt.figure(figsize=(12, 6))
sns.barplot(x=dayofweek_avg.index, y=dayofweek_avg.values, color='orange')
plt.title('Average Blood Sugar per Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Blood Sugar (mg/dL)')
plt.show()

# Hour of day averages
df['HourOfDay'] = df['Timestamp'].dt.hour
hourofday_avg = df.groupby('HourOfDay')['Measurement (mg/dL)'].mean()
plt.figure(figsize=(12, 6))
sns.barplot(x=hourofday_avg.index, y=hourofday_avg.values, color='purple')
plt.title('Average Blood Sugar per Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Blood Sugar (mg/dL)')
plt.show()

# ----------------------------------------------------------------------------
# Step 4: Match events (workout vs non‑workout) and compute statistics
# ----------------------------------------------------------------------------
# Collect glucose readings during each workout
glucose_during_workout_readings = []
for _, row in workout_df_filtered.iterrows():
    mask = (df['Timestamp'] >= row['startDate']) & (df['Timestamp'] <= row['endDate'])
    glucose_vals = df.loc[mask, 'Measurement (mg/dL)'].dropna()
    glucose_during_workout_readings.extend(glucose_vals.tolist())

# Convert to numpy array for later calculations
glucose_during_workout_readings = np.array(glucose_during_workout_readings)

# Create a set of workout timestamps for fast filtering (optional)
workout_times = set()
for _, row in workout_df_filtered.iterrows():
    workout_times.add(row['startDate'])
    workout_times.add(row['endDate'])

# For matched non‑workout readings, we need day‑of‑week (0=Monday) and hour
df['DayOfWeekNum'] = df['Timestamp'].dt.dayofweek   # Monday=0, Sunday=6
df['Hour'] = df['Timestamp'].dt.hour

# Also add same columns to workout dataframe
workout_df_filtered['DayOfWeekNum'] = workout_df_filtered['startDate'].dt.dayofweek
workout_df_filtered['Hour'] = workout_df_filtered['startDate'].dt.hour

# Collect matched non‑workout readings (same DOW and hour, but not during workout)
matched_non_workout_readings = []
for _, workout in workout_df_filtered.iterrows():
    # Find CGM readings with same day‑of‑week and same hour, but outside workout interval
    same_dow_hour = (df['DayOfWeekNum'] == workout['DayOfWeekNum']) & (df['Hour'] == workout['Hour'])
    outside_workout = ~((df['Timestamp'] >= workout['startDate']) & (df['Timestamp'] <= workout['endDate']))
    mask = same_dow_hour & outside_workout
    vals = df.loc[mask, 'Measurement (mg/dL)'].dropna().tolist()
    matched_non_workout_readings.extend(vals)

matched_non_workout_readings = np.array(matched_non_workout_readings)

# ----------------------------------------------------------------------------
# Step 5: Statistical comparison and visualisation
# ----------------------------------------------------------------------------
# Calculate averages
avg_workout = np.mean(glucose_during_workout_readings) if len(glucose_during_workout_readings) > 0 else np.nan
avg_non_workout = np.mean(matched_non_workout_readings) if len(matched_non_workout_readings) > 0 else np.nan

print("Average Glucose During Workouts         :", avg_workout)
print("Average Glucose During Matched Non‑Workout:", avg_non_workout)

# Perform t‑test (independent, Welch's t‑test because variances may differ)
if len(glucose_during_workout_readings) > 1 and len(matched_non_workout_readings) > 1:
    t_stat, p_value = stats.ttest_ind(glucose_during_workout_readings,
                                      matched_non_workout_readings,
                                      equal_var=False)
    print("T‑test results -- T‑Statistic: {:.3f}, P‑value: {:.4f}".format(t_stat, p_value))
    if p_value < 0.05:
        print("--> Statistically significant difference found!")
    else:
        print("--> No statistically significant difference.")
else:
    print("Insufficient data for t‑test.")

# Bar plot with error bars (standard deviation)
means = [avg_workout, avg_non_workout]
std_devs = [np.std(glucose_during_workout_readings, ddof=1) if len(glucose_during_workout_readings) > 1 else 0,
            np.std(matched_non_workout_readings, ddof=1) if len(matched_non_workout_readings) > 1 else 0]

plt.figure(figsize=(8,6))
plt.bar(['During Workouts', 'Matched Non‑Workout Periods'], means, yerr=std_devs, capsize=10, color=['red', 'green'])
plt.ylabel('Average Blood Glucose (mg/dL)')
plt.title('Blood Glucose Levels: Workouts vs Matched Non‑Workout Periods')
plt.show()

# ----------------------------------------------------------------------------
# Optional: ECG data parser (uncomment and adapt if you have ECG CSVs)
# ----------------------------------------------------------------------------
# path_ecg = 'electrocardiograms'
# if os.path.exists(path_ecg):
#     files = [f for f in os.listdir(path_ecg) if f.endswith('.csv')]
#     dfs = []
#     for f in files:
#         df_ecg = pd.read_csv(os.path.join(path_ecg, f), nrows=8, header=None, index_col=0, encoding='ISO-8859-1')
#         df_ecg = df_ecg.transpose()
#         df_ecg['Filename'] = f
#         dfs.append(df_ecg)
#     output = pd.concat(dfs)
#     output = output.iloc[:, :9]
#     output.columns = ['Name', 'Date of Birth', 'Recorded Date', 'Classification', 'Symptoms',
#                       'Software Version', 'Device', 'Sample Rate', 'Filename']
#     output['Recorded Date'] = pd.to_datetime(output['Recorded Date'])
#     output = output.sort_values(by='Recorded Date')
#     # Merge with glucose data to see if ECG classifications correlate with low glucose
#     # (implementation left as an exercise)

# ----------------------------------------------------------------------------
# Bonus: Hypoglycemia risk by hour of day (actionable insight)
# ----------------------------------------------------------------------------
df['IsLow'] = df['Measurement (mg/dL)'] < 70   # mg/dL threshold for low
low_by_hour = df.groupby('Hour')['IsLow'].mean() * 100
plt.figure(figsize=(12,6))
plt.bar(low_by_hour.index, low_by_hour.values, color='crimson')
plt.title('Percentage of Low Glucose Readings (<70 mg/dL) by Hour of Day')
plt.xlabel('Hour of Day (0–23)')
plt.ylabel('Low Glucose Frequency (%)')
plt.xticks(range(0,24))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Print actionable advice based on the t‑test result
if 'p_value' in locals() and p_value < 0.05 and avg_workout < avg_non_workout:
    print("\n🔔 ACTIONABLE ADVICE:")
    print("Your glucose drops significantly during workouts. Try one of these before exercising:")
    print(" - Eat 15–20g fast‑acting carbohydrates without insulin.")
    print(" - Reduce active insulin (e.g., lower basal rate 60 min prior).")
    print(" - Choose shorter or lower‑intensity workouts, or schedule them when glucose is >150 mg/dL.")
elif 'p_value' in locals() and p_value < 0.05 and avg_workout > avg_non_workout:
    print("\n🔔 ACTIONABLE ADVICE:")
    print("Your glucose rises during workouts – an unusual stress response. Consider:")
    print(" - A small bolus (0.5–1 unit) 15 min before starting.")
    print(" - Cooling down with light walking after the main workout.")
    print(" - Discussing this pattern with your endocrinologist.")
else:
    print("\n✅ Your glucose response to exercise appears stable. Monitor regularly and trust your routine.")
