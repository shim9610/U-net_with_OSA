import torch
import numpy as np
import torch.nn.functional as F
from model_utils import SpectrumModel, SpectrumDataset, create_spectrum, get_model_size, min_max_normalize
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# 임의의 데이터 1개 생성 함수
def custom_MSE(output, target,epsilon=1e-10,max_value=1e10):
        # 비율적인 차이를 계산
    ratio_diff = (target - output) / (target + epsilon)
    
    # NaN 및 무한대를 제거
    ratio_diff = torch.where(torch.isfinite(ratio_diff), ratio_diff, torch.zeros_like(ratio_diff))
    ratio_diff = torch.clamp(ratio_diff, -max_value, max_value)
    # 비율 차이의 제곱 평균 계산
    loss = torch.sum(torch.abs(ratio_diff))
    #print(loss)
    return loss

# 모델 로드
model = torch.load('logUnet_best_model_100kresi_BC.pth', map_location=torch.device('cpu'))
model.eval()


data = []
input, sol =  create_spectrum(solution_type='dual')
data.append((torch.tensor(input), torch.tensor(sol)))
dataset = SpectrumDataset(data)
dataset=DataLoader(dataset, batch_size=32, shuffle=True)
Truedata=[]
predicted_params=[]
sol=[]
ori=[]
for data, target in dataset:
    ori=data
    
    with torch.no_grad():
        predicted_result = model(data,mode='train_dual')
        
    sol=(target)

    ori,no=min_max_normalize(data)
loss=custom_MSE(predicted_result,sol)
print(loss)
X = np.linspace(669.46, 670.544, 160)+0.8
X2= np.linspace(669.46, 670.544, 160)+0.8
plt.clf()
plt.rcParams.update({'font.size': 34})

plt.plot(X2, predicted_result[:,0,:].squeeze().numpy(), label='Predicted emission')
plt.plot(X2, sol[:,0,:].squeeze().numpy(), label='Real emission')
plt.plot(X, ori.squeeze().numpy(), label='Input Data')
plt.plot(X2, predicted_result[:,1,:].squeeze().numpy(), label='Predicted absorption')
plt.plot(X2, sol[:,1,:].squeeze().numpy(), label='Real Absorption')
plt.plot(X2, ori.squeeze().numpy()/predicted_result[:,1,:].squeeze().numpy(), label='absorption cal')
plt.legend(fontsize=15)
plt.show()

print(no.clone().detach())

