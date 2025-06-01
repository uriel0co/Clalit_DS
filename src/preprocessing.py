import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import joblib
import os
from sklearn.utils import resample

def preprocess(df, model_dir="models"):
    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # 1. Remove Age Outliers
    df = df[(df['Age'] >= 0) & (df['Age'] <= 120)].copy()
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # 2. Exclude the Sex Column
    if 'Sex' in df.columns:
        df = df.drop(columns=['Sex'])

    # 3. Upsample Autoimmune Disorder Cases (based on avg len of other diseases)
    if 'Disease' in df.columns:
        disease_counts = df['Disease'].value_counts()
        if 'Autoimmune Disorder' in disease_counts:
            other_diseases = disease_counts.drop('Autoimmune Disorder')
            avg_other = int(other_diseases.mean())
            minority = df[df['Disease'] == 'Autoimmune Disorder']
            if len(minority) > 0 and avg_other > 0:
                minority_upsampled = resample(
                    minority,
                    replace=True,
                    n_samples=avg_other,
                    random_state=42
                )
                majority = df[df['Disease'] != 'Autoimmune Disorder']
                df = pd.concat([majority, minority_upsampled])
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 4. Normalize Autoimmune Disorder BMI Values from 0 to 10
    if 'Disease' in df.columns and 'BMI' in df.columns:
        mask = df['Disease'] == 'Autoimmune Disorder'
        scaler_bmi = MinMaxScaler(feature_range=(0, 10))
        df.loc[mask, 'BMI'] = scaler_bmi.fit_transform(df.loc[mask, ['BMI']])
        joblib.dump(scaler_bmi, os.path.join(model_dir, "autoimmune_bmi_minmax_scaler.pkl"))

    # Encode categorical features (excluding 'Sex' as it's dropped)
    cat_cols = ['Family History of Disease', 'Medication Use', 'Occupation Type']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['Disease'])

    # Save label encoder
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))

    # Store the class names for further use
    class_names = list(le.classes_)

    # Drop ID and target from features
    X = df.drop(['ID', 'Disease'], axis=1)

    # Store feature names after preprocessing
    feature_names = list(X.columns)

    # Scaling (StandardScaler for all features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print(X_scaled)
    return X_scaled, y, le, scaler, feature_names, class_names