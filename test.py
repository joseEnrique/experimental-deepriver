from river import compose, preprocessing, metrics, evaluate
import time
from datetime import datetime as dt
from deep_river.regression import RollingRegressor

import torch
from river.datasets import synth
from tqdm import tqdm

from lstm import NewLstmModule

_ = torch.manual_seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = synth.FriedmanDrift(
   drift_type='lea',
   position=(2000, 5000, 8000),
   seed=123
)

metric = metrics.MAE()


ae = RollingRegressor(module=NewLstmModule,
    loss_fn="mse",
    optimizer_fn="adam",
    window_size=100,
    lr=1e-2,
    device="cuda:1",
    hidden_size=64,  # parameters of MyModule can be overwritten
    append_predict=False,)

processed_x = []
index = 0
start_time = time.time()
start_formatted_time = dt.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S.%f')
for x, y in (dataset.take(1000)):
    #print (f"------------------- Iteration {index} -------------------")
    #print (x)
    processed_x.append(x)
    a = ae.learn_one(x=x, y=y)
    if index != 0:
        for_x = processed_x[:-1]
    windows_x = ae.get_windows()
    lenght_windows_x = len(windows_x)
    line_to_writ = ""
    if lenght_windows_x > 0:
        line_to_writ += "1st: "+str(windows_x[0][0])+" "
    if lenght_windows_x > 1:
        line_to_writ += "2nd: "+ str(windows_x[1][0])
    #line_to_writ += " Length of the windows: "+str(lenght_windows_x)
    #print (line_to_writ)

    y_pred = ae.predict_one(x)
    metric.update(y_true=y, y_pred=y_pred)
    index += 1


print (f'Final metric: {metric.get()}')