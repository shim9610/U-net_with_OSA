import warnings
import logging
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import threading
import time
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model_utils import SpectrumModel, SpectrumDataset, create_spectrum, get_model_size
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

first_predicted_params = None
first_target = None
first_predicted_params_ch2 = None
first_target_ch2 = None
train_losses = []
val_losses = []
def run_dash_app():
    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id='live-graph', animate=False),
        dcc.Graph(id='params-graph', animate=False),
        dcc.Graph(id='params-graph_ch2', animate=False),
        dcc.Interval(
            id='interval-component',
            interval=1*10,  # in milliseconds
            n_intervals=0
        )
    ])

    @app.callback(Output('live-graph', 'figure'),
                  [Input('interval-component', 'n_intervals')])
    def update_graph(n):
        if not train_losses or not val_losses: 
            return go.Figure()

        x = list(range(len(train_losses)))
        train_y = (train_losses) 
        val_y = (val_losses)
        train_data = go.Scatter(
            x=x,
            y=train_y,
            name='Train Loss',
            mode='lines+markers',
        )
        val_data = go.Scatter(
            x=x,
            y=val_y,
            name='Validation Loss',
            mode='lines+markers',
        )
        
        return {
            'data': [train_data, val_data],
            'layout': go.Layout(
                xaxis={'autorange': True},
                yaxis={'type': 'log', 'autorange': True}, 
                #yaxis={'autorange': True},  
                title="Training and Validation Loss Over Time"
            )
        }
    @app.callback(Output('params-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
    def update_params_graph(n):
        if first_predicted_params is None or first_target is None:
            #print("No data to plot yet.")
            return go.Figure()
        # Flatten the tensors and convert to numpy arrays
        pred_data = first_predicted_params.numpy().flatten()
        target_data = first_target.numpy().flatten()

        pred_plot = go.Scatter(
            x=list(range(len(pred_data))),
            y=pred_data,
            name='Predicted Params (First Batch)',
            mode='lines+markers',
            marker=dict(color='red')
        )
        target_plot = go.Scatter(
            x=list(range(len(target_data))),
            y=target_data,
            name='Target Params (First Batch)',
            mode='lines+markers',
            marker=dict(color='blue')
        )
        
        return {
            'data': [pred_plot, target_plot],
            'layout': go.Layout(
                xaxis={'autorange': True},
                yaxis={'autorange': True},
                title="Predicted vs Target Params (First Batch)"
            )
        }
    @app.callback(Output('params-graph_ch2', 'figure'),
              [Input('interval-component', 'n_intervals')])
    def update_params_graph(n):
        if first_predicted_params_ch2 is None or first_target_ch2 is None:
            return go.Figure()
        # Flatten the tensors and convert to numpy arrays
        pred_data = first_predicted_params_ch2.numpy().flatten()
        target_data = first_target_ch2.numpy().flatten()
        pred_plot = go.Scatter(
            x=list(range(len(pred_data))),
            y=pred_data,
            name='Predicted Params (First Batch)',
            mode='lines+markers',
            marker=dict(color='red')
        )
        target_plot = go.Scatter(
            x=list(range(len(target_data))),
            y=target_data,
            name='Target Params (First Batch)',
            mode='lines+markers',
            marker=dict(color='blue')
        )
        
        return {
            'data': [pred_plot, target_plot],
            'layout': go.Layout(
                xaxis={'autorange': True},
                yaxis={'autorange': True},
                title="Predicted vs Target Params (First Batch)"
            )
        }
    app.run_server(debug=False, use_reloader=False)
################################################################################
#Visualisation Traning curve
################################################################################
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
train_losses=[]
val_losses=[]
losses = []
thread = threading.Thread(target=run_dash_app, args=())
thread.start()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
num_samples = 3333333
data = []
for _ in range(num_samples):
   
    input, sol =  create_spectrum(solution_type='dual')
    data.append((torch.tensor(input), torch.tensor(sol)))
dataset = SpectrumDataset(data)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True)

def custom_MSE_per(output, target,epsilon=1e-10,max_value=1e10):
    ratio_diff = (target - output) / (target + epsilon)
    ratio_diff = torch.where(torch.isfinite(ratio_diff), ratio_diff, torch.zeros_like(ratio_diff))
    ratio_diff = torch.clamp(ratio_diff, -max_value, max_value)
    loss = torch.mean(torch.abs(ratio_diff))*100
    target_channel1 = target[:, 0, :]  #
    peak_values, peak_indices = torch.max(target_channel1, dim=1)
    batch_indices = torch.arange(output.size(0), device=output.device)  #
    peak_output = output[batch_indices, 0, peak_indices]  
    peak_target = peak_values  
    peak_loss = torch.abs(peak_target - peak_output) / (peak_target + epsilon) *100
    return loss*2 + torch.mean(peak_loss)

model = SpectrumModel()
model = SpectrumModel().to(device)
#model = torch.load('logUnet_best_model_3333k_liquid512_n.pth')
#model.to(device) 
model_size = get_model_size(model)
print(model_size)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 
#criterion = nn.CrossEntropyLoss()
num_epochs = 5000

patience = 50  # 
best_val_loss = float('inf')  # 
epochs_without_improvement = 0
# 학습률 스케줄러 정의
scale=100
loss_scale=3
gradient_penalty_weight=220
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-15)
i=0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for data, target in train_dataloader:
        data, target = data.to(device), target.to(device)
        predicted_params = model(data,mode='train_dual')
        loss = custom_MSE_per(predicted_params, target)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del loss
        del data
        del target
        del predicted_params
        torch.cuda.empty_cache()  # 

    train_losses.append(train_loss / len(train_dataloader))
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}')

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in val_dataloader:
            data, target = data.to(device), target.to(device)
            predicted_params = model(data,mode='train_dual')

            
            if i == 0:
                first_predicted_params = predicted_params[0,0,:].detach().cpu().clone()
                first_target = target[0,0,:].detach().cpu().clone()
                first_predicted_params_ch2 = predicted_params[0,1,:].detach().cpu().clone()
                first_target_ch2 = target[0,1,:].detach().cpu().clone()
            i=i+1
            loss = custom_MSE_per(predicted_params, target)
            val_loss += loss.item()

            del loss
            del data
            del target
            del predicted_params
            torch.cuda.empty_cache()  
        i=0
    val_losses.append(val_loss / len(val_dataloader))
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_losses[-1]:.4f}')
    scheduler.step(val_loss)
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        epochs_without_improvement = 0
        torch.save(model, 'MUnet_best_model_3333k_liquid512.pth') 
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break
torch.save(model, 'MUnet_3333k_liquid512.pth')
print("Full model saved.")
del model
del optimizer
torch.cuda.empty_cache()