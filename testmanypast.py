import torch
import torch.nn as nn
import torch.nn.functional as F
from river.datasets import synth
from river import metrics
#from sklearn import metrics, config_context
from fluvialgen.past_forecast_batcher import PastForecastBatcher
from deep_river.regression import RegressorInitialized, RollingRegressorInitialized
import time, csv

dataset = synth.FriedmanDrift(
    drift_type='lea',
    position=(2000, 5000, 8000),
    seed=123
)

def get_activation(activation_name):
    """Returns a callable activation function given its name."""
    name = activation_name.lower()
    if name == "relu":
        return F.relu
    elif name == "tanh":
        return torch.tanh
    elif name == "sigmoid":
        return torch.sigmoid
    elif name == "linear":
        return lambda x: x
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

class RollingRegressorLstmModule(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers= 1):
        super().__init__()
        self.n_features=n_features
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False)
        self.activation = get_activation("linear")
        self.fc = nn.Linear(in_features=hidden_size,out_features=1) #Dense

    def forward(self, X, **kwargs):
        output, (hn, cn) = self.lstm(X)  # lstm with input, hidden, and internal state
        x = self.fc(output)
        x = self.activation(x)
        x =  x.squeeze(-1)
        return x


def experimentRollingRegressor (past_size,hidden_size,num_layer):
    ae = RollingRegressorInitialized(
        module=RollingRegressorLstmModule(10),
        loss_fn="mse",
        optimizer_fn="adam",
        window_size=past_size,  # We'll keep a window of the last 5 *batches*
        lr=1e-2,
        device="cuda:0",
        hidden_size=64,  # Matches init of NewLstmModule
        #append_predict=False
    )
    batcher = PastForecastBatcher(
        dataset=dataset,
        past_size=past_size,
        forecast_size=0,
        n_instances=10000
    )
    with open(f'results/pastrolling/p{past_size}-h{hidden_size}-l{num_layer}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(['iteration', 'predict_time_ms', 'learn_time_ms', 'iteration_time_ms','metrics'])
        metric = metrics.MAE()
        for i,(x, y,current_x,current_y)  in enumerate(batcher):
            iteration_start = time.perf_counter()
            #ae.predict_many(X=x).values
            start_predict = time.perf_counter()
            y_pred = ae.predict_one(current_x)
            end_predict = time.perf_counter()
            predict_time_ms = (end_predict - start_predict) * 1000
            start_learn = time.perf_counter()
            #print(x.shape)
            #print(x)
            #print(y)
            ae.learn_many(X=x, y=y)
            end_learn = time.perf_counter()
            learn_time_ms = (end_learn - start_learn) * 1000
            iteration_end = time.perf_counter()
            iteration_time_ms = (iteration_end - iteration_start) * 1000
            metric.update(current_y, y_pred)
            writer.writerow([i, predict_time_ms, learn_time_ms, iteration_time_ms,metric.get()])


#batch_sizes = [10,20,30,40,50,100,200]#instance_sizes = [10,20,30,40,50,100,200]
past_sizes = [600]
hidden_sizes = [64]
num_layers = [1]
for past_size in past_sizes:

    for hidden_size in hidden_sizes:
        for num_layer in num_layers:
            iteration_start = time.perf_counter()
            experimentRollingRegressor(past_size, hidden_size, num_layer)
            end_predict = time.perf_counter()
            predict_time_ms = (end_predict - iteration_start) * 1000
            print("-----")
            print(past_size,hidden_size,num_layer)
            print(f"Tiempo total de predicción: {predict_time_ms} ms")
            print ("-----")

