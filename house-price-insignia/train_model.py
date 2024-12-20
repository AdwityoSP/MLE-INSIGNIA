import pandas as pd
import re
from sklearn.linear_model import LassoLars
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')

data = {
    'houseId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'length': [20, 40, 3000, 1000, 20, 50, 20, 50, 20, 10],
    'lengthUnit': ['meter', 'meter', 'centimeter', 'centimeter', 'meter', 'meter', 'meter', 'meter', 'meter', 'meter'],
    'width': [10, 20, 2000, 3000, 50, 10, 20, 20, 30, 20],
    'widthUnit': ['meter', 'meter', 'centimeter', 'centimeter', 'meter', 'meter', 'meter', 'meter', 'meter', 'meter'],
    'isCarport': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'price': ['5 Billion', '18 Billion', '13 Billion', '6 Billion', '21 Billion', '11 Billion', '8 Billion',
              'IDR 15 Billion', 'IDR 13 Billion', 'IDR 11 Billion'],
    'notes': ['TRAINING DATA', 'TRAINING DATA', 'TRAINING DATA', 'TRAINING DATA',
              'TRAINING DATA', 'TEST DATA', 'TEST DATA',
              'VALIDATION DATA', 'VALIDATION DATA', 'VALIDATION DATA']
}

df = pd.DataFrame(data)

def convert_price(p_str):
    match = re.search(r'(\d+(\.\d+)?)', p_str)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Cannot parse price from '{p_str}'")

df['price_val'] = df['price'].apply(convert_price)

def to_meters(value, unit):
    if unit == 'meter':
        return value
    elif unit == 'centimeter':
        return value / 100.0
    else:
        raise ValueError(f"Unknown unit: {unit}")

df['length_m'] = [to_meters(l, u) for l, u in zip(df['length'], df['lengthUnit'])]
df['width_m'] = [to_meters(w, u) for w, u in zip(df['width'], df['widthUnit'])]

train_df = df[df['notes'] == 'TRAINING DATA'].copy()
test_df = df[df['notes'] == 'TEST DATA'].copy()
validation_df = df[df['notes'] == 'VALIDATION DATA'].copy()

feature_columns = ['length_m', 'width_m', 'isCarport']

X_train = train_df[feature_columns]
y_train = train_df['price_val']

X_test = test_df[feature_columns]
y_test = test_df['price_val']

X_val = validation_df[feature_columns]
y_val = validation_df['price_val']

lasso_lars = LassoLars(alpha=0.01)
lasso_lars.fit(X_train, y_train)

y_pred_test = lasso_lars.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = mean_squared_error(y_test, y_pred_test)

print("Test Set Evaluation:")
print(f"R-squared: {r2_test:.2f}")
print(f"RMSE: {rmse_test:.2f} Billion")

y_pred_val = lasso_lars.predict(X_val)
r2_val = r2_score(y_val, y_pred_val)
rmse_val = mean_squared_error(y_val, y_pred_val)

print("\nValidation Set Evaluation:")
print(f"R-squared: {r2_val:.2f}")
print(f"RMSE: {rmse_val:.2f} Billion")

model_filename = 'lasso_lars_model.pkl'
joblib.dump(lasso_lars, model_filename)
print(f"\nModel saved to {model_filename}")
