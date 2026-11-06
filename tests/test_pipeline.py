import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.modeling import build_pipeline, evaluate, map_gender, ORGANIZATIONAL

def test_gender_mapping():
    assert map_gender("M") == "Masculino (autodeclarado)"
    assert map_gender("Female") == "Feminino (autodeclarado)"
    assert map_gender("genderqueer") == "Outra resposta/ambígua"

def test_pipeline_handles_missing_and_new_categories():
    cols = ["no_employees", "remote_work"]
    x = pd.DataFrame({"no_employees":["1-5","6-25","6-25","1-5"],"remote_work":["Yes","No",None,"No"]})
    y = [1,0,1,0]
    model=build_pipeline(LogisticRegression(max_iter=1000),cols)
    model.fit(x,y)
    test=pd.DataFrame({"no_employees":["10000+"],"remote_work":[None]})
    assert model.predict_proba(test).shape == (1,2)

def test_metrics():
    m=evaluate(np.array([0,0,1,1]),np.array([0.1,0.6,0.7,0.9]))
    assert (m["TN"],m["FP"],m["FN"],m["TP"]) == (1,1,0,2)
