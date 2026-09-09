import joblib

xgb = joblib.load('model/xgboost.pkl')
joblib.dump(xgb, 'model/best_model.pkl')
joblib.dump('XGBoost', 'model/best_model_name.pkl')
print('Best model updated to XGBoost.')