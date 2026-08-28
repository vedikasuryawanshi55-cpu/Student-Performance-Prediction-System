import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------
# 1. Load Dataset
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "analytics" / "student_performance.csv"

df = pd.read_csv(CSV_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------
# 2. Select Features and Target
# --------------------------------

X = df[
    [
        "StudyHours",
        "Attendance",
        "PreviousScore",
        "AssignmentScore",
        "SleepHours"
    ]
]

y = df["FinalScore"]


# --------------------------------
# 3. Split Dataset
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------
# 4. Linear Regression
# --------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_prediction = linear_model.predict(X_test)


linear_mae = mean_absolute_error(
    y_test,
    linear_prediction
)

linear_mse = mean_squared_error(
    y_test,
    linear_prediction
)

linear_r2 = r2_score(
    y_test,
    linear_prediction
)


print("\n--- Linear Regression ---")

print("MAE:", linear_mae)

print("MSE:", linear_mse)

print("R2 Score:", linear_r2)


# --------------------------------
# 5. Random Forest
# --------------------------------

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)

rf_prediction = random_forest.predict(X_test)


rf_mae = mean_absolute_error(
    y_test,
    rf_prediction
)

rf_mse = mean_squared_error(
    y_test,
    rf_prediction
)

rf_r2 = r2_score(
    y_test,
    rf_prediction
)


print("\n--- Random Forest ---")

print("MAE:", rf_mae)

print("MSE:", rf_mse)

print("R2 Score:", rf_r2)


# --------------------------------
# 6. Compare Models
# --------------------------------

print("\n--- Model Comparison ---")

if rf_r2 > linear_r2:

    final_model = random_forest

    print("Random Forest performed better.")

else:

    final_model = linear_model

    print("Linear Regression performed better.")


# --------------------------------
# 7. Save Final Model
# --------------------------------

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "student_model.pkl"

joblib.dump(final_model, MODEL_PATH)

print("\nFinal model saved successfully!")

print("Model location:", MODEL_PATH)