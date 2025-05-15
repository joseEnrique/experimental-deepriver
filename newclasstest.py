import torch
import torch.nn as nn
import torch.nn.functional as F
from river.datasets import synth
from river import metrics
#from sklearn import metrics, config_context
from fluvialgen.movingwindow_generator import MovingWindowBatcher
from deep_river.regression import RegressorInitialized, RollingRegressorInitialized
import time, csv
from pprint import pprint
import cupy as cp

from deep_river.regression.dual_device_regressor import DualGPURegressorInitialized

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
class RegressorLstmModule(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers= 1):
        super().__init__()
        self.n_features=n_features
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,  num_layers=num_layers, bidirectional=False)
        self.activation = get_activation("linear")
        self.fc = nn.Linear(in_features=hidden_size,out_features=1) #Dense

    def forward(self, X, **kwargs):
        output, (hn, cn) = self.lstm(X)  # lstm with input, hidden, and internal state
        x = self.fc(output)
        x = self.activation(x)
        #x =  x.squeeze(-1)
        return x


def experimentDualDeviceRollingRegressor (batch_size,instance_size,hidden_size,num_layer):
    ae = DualGPURegressorInitialized(
        module=RollingRegressorLstmModule(10),
        loss_fn="mse",
        optimizer_fn="adam",
        window_size=batch_size*instance_size,  # We'll keep a window of the last 5 *batches*
        lr=1e-2,
        train_device="cuda:0",
        inference_device="cuda:1",
        hidden_size=64,  # Matches init of NewLstmModule
        #append_predict=False
    )
    batcher = MovingWindowBatcher(
        dataset=dataset,
        instance_size=instance_size,
        batch_size=batch_size,
        n_instances=10000
    )
    metric = metrics.MAE()
    with open(f'results/dualdevicerollingregressor/b{batch_size}-i{instance_size}-h{hidden_size}-l{num_layer}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(['iteration', 'predict_time_ms', 'learn_time_ms', 'iteration_time_ms','metrics'])

        for i,(x, y)  in enumerate(batcher):
            y_trues = []
            y_preds = []
            iteration_start = time.perf_counter()
            #ae.predict_many(X=x).values
            y_trues.extend(y)
            start_predict = time.perf_counter()
            y_preds.extend(ae.predict_many(X=x).values)
            end_predict = time.perf_counter()
            predict_time_ms = (end_predict - start_predict) * 1000

            start_learn = time.perf_counter()
            ae.learn_many(X=x, y=y)
            end_learn = time.perf_counter()
            learn_time_ms = (end_learn - start_learn) * 1000
            iteration_end = time.perf_counter()
            iteration_time_ms = (iteration_end - iteration_start) * 1000

            for ypred, ytrue in zip(y_preds, y_trues):
                metric.update(ytrue, ypred[0])
            writer.writerow([i, predict_time_ms, learn_time_ms, iteration_time_ms,metric.get()])





#batch_sizes = [10,20,30,40,50,100,200]
#instance_sizes = [10,20,30,40,50,100,200]
batch_sizes = [60,100]
instance_sizes = [100,60]
hidden_sizes = [64]
num_layers = [1]
for batch_size in batch_sizes:
    for instance_size in instance_sizes:
        for hidden_size in hidden_sizes:
            for num_layer in num_layers:
                print(batch_size, instance_size,hidden_size,num_layer)
                # Usamos un ThreadPoolExecutor para ejecutar ambas funciones concurrentemente.
                iteration_start = time.perf_counter()
                experimentDualDeviceRollingRegressor(batch_size,instance_size,hidden_size,num_layer)
                end_predict = time.perf_counter()
                predict_time_ms = (end_predict - iteration_start) * 1000
                print("-----")
                print(batch_size, instance_size, hidden_size, num_layer)
                print(f"Tiempo total de predicción: {predict_time_ms} ms")
                print("-----")