from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

def analyze_churn(df, selected_features):
    """
    Analyzes churn. If a target variable like 'Churn' exists, trains an ML model.
    Otherwise, computes an engineering-based Churn Risk Score.
    """
    # Check for common historical churn column names (case-insensitive)
    possible_target_names = ['churn', 'exited', 'left', 'is_churn']
    target_col = None
    
    for col in df.columns:
        if col.lower() in possible_target_names:
            target_col = col
            break

    # MODE A: Machine Learning Classification (If data contains historical targets)
    if target_col:
        X = df[selected_features]
        y = df[target_col]
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_split_size=0.2, random_state=42)
        
        # Train Random Forest Classifier
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Map risk probabilities back to the entire dataframe
        df['Churn_Probability'] = model.predict_proba(X)[:, 1]
        df['Churn_Risk_Level'] = np.where(df['Churn_Probability'] > 0.5, 'High Risk', 'Low Risk')
        
        return df, f"✅ ML Model Trained! Test Accuracy: {acc:.2%}"

    # MODE B: Heuristic Risk Scoring (For Standard Segmentation datasets like Mall Customers)
    else:
        # Let's infer risk mathematically: Lower spending score + lower engagement columns = higher risk
        # We normalize features to a 0-1 scale to calculate a risk index safely
        risk_score = np.zeros(len(df))
        
        # Find spending or income columns dynamically to build the risk rule
        spending_col = [c for c in df.columns if 'spend' in c.lower()]
        
        if spending_col:
            # Lower spending score = Higher risk of churn
            max_val = df[spending_col[0]].max()
            min_val = df[spending_col[0]].min()
            normalized_spend = (df[spending_col[0]] - min_val) / (max_val - min_val)
            risk_score += (1 - normalized_spend) * 0.7  # 70% weight on low spending habits
            
        # Add a placeholder factor for age/variance to flesh out the metric
        age_col = [c for c in df.columns if 'age' in c.lower()]
        if age_col:
            max_age = df[age_col[0]].max()
            normalized_age = df[age_col[0]] / max_age
            risk_score += (normalized_age) * 0.3  # 30% weight balancing older demographic passivity
            
        df['Churn_Probability'] = np.clip(risk_score, 0, 1)
        df['Churn_Risk_Level'] = np.where(df['Churn_Probability'] > 0.6, 'High Risk', 
                                          np.where(df['Churn_Probability'] > 0.3, 'Medium Risk', 'Low Risk'))
        
        return df, "⚠️ No historical churn column detected. Computed Churn Risk Engine via feature behaviors."