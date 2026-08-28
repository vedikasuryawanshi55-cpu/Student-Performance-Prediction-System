import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# 1. Load the dataset
# --------------------------------

df = pd.read_csv("analytics/student_performance.csv")

print("Dataset loaded successfully!")

# --------------------------------
# 2. Display first 5 rows
# --------------------------------

print("\nFirst 5 rows:")
print(df.head())

# --------------------------------
# 3. Dataset shape
# --------------------------------

print("\nDataset Shape:")
print(df.shape)

# --------------------------------
# 4. Information about dataset
# --------------------------------

print("\nDataset Information:")
df.info()

# --------------------------------
# 5. Statistical summary
# --------------------------------

print("\nStatistical Summary:")
print(df.describe())

# --------------------------------
# 6. Check missing values
# --------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# --------------------------------
# 7. Check duplicate rows
# --------------------------------

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# --------------------------------
# 8. Study Hours vs Final Score
# --------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["StudyHours"],
    df["FinalScore"]
)

plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.title("Study Hours vs Final Score")

plt.show()

# --------------------------------
# 9. Attendance vs Final Score
# --------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Attendance"],
    df["FinalScore"]
)

plt.xlabel("Attendance")
plt.ylabel("Final Score")
plt.title("Attendance vs Final Score")

plt.show()

# --------------------------------
# 10. Previous Score vs Final Score
# --------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["PreviousScore"],
    df["FinalScore"]
)

plt.xlabel("Previous Score")
plt.ylabel("Final Score")
plt.title("Previous Score vs Final Score")

plt.show()

# --------------------------------
# 11. Correlation Matrix
# --------------------------------

correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()

print("\nEDA completed successfully!")