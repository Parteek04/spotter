import numpy as np,pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor
DATA=Path("data")
def f(df):
 x=df.copy(); dt=pd.to_datetime(x.date)
 x["year"]=dt.dt.year; x["month"]=dt.dt.month; x["day"]=dt.dt.day; x["dayofweek"]=dt.dt.dayofweek; x["dayofyear"]=dt.dt.dayofyear; x["weekofyear"]=dt.dt.isocalendar().week.astype(int)
 x["doy_sin"]=np.sin(2*np.pi*x.dayofyear/365.25); x["doy_cos"]=np.cos(2*np.pi*x.dayofyear/365.25); x["route"]=x.pickup.astype(str)+"__"+x.delivery.astype(str); x["distance_log"]=np.log1p(x.distance); x["weight_log"]=np.log1p(x.weight)
 return x.drop(columns=["posted_rate","load_id","date"],errors="ignore")
train=pd.read_csv(DATA/"train-test.csv"); val=pd.read_csv(DATA/"validation.csv"); template=pd.read_csv(DATA/"validation-predictions-template.csv")
X=f(train); Xv=f(val); y=train.posted_rate.astype(float); cats=[c for c in X.columns if X[c].dtype=="object"]; ps=[]
for seed in [42,7,2026]:
 m=CatBoostRegressor(iterations=500,depth=6,learning_rate=.03,l2_leaf_reg=50,loss_function="RMSE",random_seed=seed,verbose=False); m.fit(X,y,cat_features=cats); ps.append(m.predict(Xv))
o=template.copy(); o["predicted_rate"]=np.maximum(np.mean(ps,axis=0),.01); o.to_csv("validation_predictions.csv",index=False); print("Saved validation_predictions.csv")
