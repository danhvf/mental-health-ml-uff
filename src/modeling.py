"""Experimentos reproduzíveis, OSMI 2014. Sem imputação/encoding fora do pipeline."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, roc_curve)

SEED = 42
TARGET = 'treatment'
ORGANIZATIONAL = [
    'self_employed', 'no_employees', 'remote_work', 'tech_company',
    'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity',
    'leave', 'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_health_interview',
    'phys_health_interview', 'mental_vs_physical', 'obs_consequence'
]
INDIVIDUAL = ['Age', 'Gender_group', 'family_history', 'work_interfere']
FULL = ORGANIZATIONAL + INDIVIDUAL
FULL_WITHOUT_INTERFERENCE = ORGANIZATIONAL + [f for f in INDIVIDUAL if f != 'work_interfere']
MALE = {'m','male','man','cis male','cis man','male (cis)','mail','mal','maile','malr','msle','make'}
FEMALE = {'f','female','woman','cis female','female (cis)','cis-female/femme','femake','femail',
          'female ','trans-female','trans woman','female (trans)'}

def map_gender(v):
    if pd.isna(v): return 'Não declarado/outra resposta'
    x = re.sub(r'\s+', ' ', str(v).strip().lower())
    if x in MALE: return 'Masculino (autodeclarado)'
    if x in FEMALE: return 'Feminino (autodeclarado)'
    return 'Outra resposta/ambígua'

def load_dataset(path='data/survey.csv'):
    df = pd.read_csv(path)
    assert TARGET in df.columns, 'Coluna treatment ausente.'
    if set(df[TARGET].dropna().unique()) != {'Yes','No'}:
        raise ValueError('Target diferente de Yes/No.')
    df = df.copy()
    df['Gender_group'] = df['Gender'].map(map_gender)
    # Não excluir respondentes; preservar como ausente para imputação dentro da CV.
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').where(lambda x: x.between(18,90), np.nan)
    return df

def build_pipeline(estimator, columns):
    numeric = [c for c in columns if c == 'Age']
    categorical = [c for c in columns if c != 'Age']
    steps = []
    if numeric:
        steps.append(('numericas', Pipeline([
            ('imputar', SimpleImputer(strategy='median')),
            ('normalizar', StandardScaler())
        ]), numeric))
    steps.append(('categoricas', Pipeline([
        ('imputar', SimpleImputer(strategy='constant', fill_value='Não informado')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ]), categorical))
    ct = ColumnTransformer(steps, remainder='drop')
    return Pipeline([('preprocessamento',ct), ('classificador',estimator)])

def candidates(columns):
    return {
     'Regressão logística': (
       build_pipeline(LogisticRegression(max_iter=2000, random_state=SEED), columns),
       {'classificador__C': [0.1, 1.0, 10.0], 'classificador__class_weight': [None]}),
     'Árvore de decisão': (
       build_pipeline(DecisionTreeClassifier(random_state=SEED), columns),
       {'classificador__max_depth':[3,5,8], 'classificador__min_samples_leaf':[5]}),
     'Random Forest': (
       build_pipeline(RandomForestClassifier(n_estimators=180, random_state=SEED,n_jobs=1), columns),
       {'classificador__max_depth':[6,12,None], 'classificador__min_samples_leaf':[2]})
    }

def evaluate(y_true, probabilities, threshold=0.5):
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    return {
       'AUROC':float(roc_auc_score(y_true, probabilities)),
       'AUPRC':float(average_precision_score(y_true, probabilities)),
       'Acurácia':float(accuracy_score(y_true,pred)),
       'Precisão':float(precision_score(y_true,pred,zero_division=0)),
       'Recall':float(recall_score(y_true,pred,zero_division=0)),
       'F1':float(f1_score(y_true,pred,zero_division=0)),
       'TN':int(confusion_matrix(y_true,pred,labels=[0,1])[0,0]),
       'FP':int(confusion_matrix(y_true,pred,labels=[0,1])[0,1]),
       'FN':int(confusion_matrix(y_true,pred,labels=[0,1])[1,0]),
       'TP':int(confusion_matrix(y_true,pred,labels=[0,1])[1,1])
    }

def bootstrap_auc(y, prob, n_boot=1000, seed=SEED):
    y=np.asarray(y);prob=np.asarray(prob); rng=np.random.default_rng(seed); vals=[]
    for _ in range(n_boot):
        idx=rng.integers(0,len(y),len(y))
        if len(set(y[idx]))>1:
            vals.append(roc_auc_score(y[idx],prob[idx]))
    return [float(v) for v in np.percentile(vals,[2.5,97.5])]

def bootstrap_auc_diff(y,p1,p0,n_boot=1000,seed=SEED):
    y=np.asarray(y);p1=np.asarray(p1);p0=np.asarray(p0); rng=np.random.default_rng(seed); vals=[]
    for _ in range(n_boot):
        idx=rng.integers(0,len(y),len(y))
        if len(set(y[idx]))>1:
            vals.append(roc_auc_score(y[idx],p1[idx])-roc_auc_score(y[idx],p0[idx]))
    return [float(v) for v in np.percentile(vals,[2.5,97.5])]
