import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import VoigtClass as VC
import random
from VoigtClass import TransitionLine
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.signal import savgol_filter
import torch.distributions as dist 

class SUDOVoigtLayer(nn.Module):
    def __init__(self):
        super(SUDOVoigtLayer, self).__init__()

    def forward(self, x, size):
        # FC 레이어의 출력 x는 [배치 크기, 3] 형태라고 가정합니다.
        # 이 x를 mu, gamma, sigma로 분리합니다.
        mu, gamma, sigma, amp = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        # gamma가 너무 작지 않도록 epsilon 값을 더해줍니다.
        epsilon = 1e-5
        gamma = torch.clamp(gamma, min=epsilon)
        sigma = torch.clamp(sigma, min=epsilon)
        gamma=gamma*size
        sigma=sigma*size
        mu=mu*size
        # x를 0에서 size까지의 값으로 생성하고, 배치 크기에 맞게 확장합니다.
        linspace_x = torch.linspace(0, size, steps=size).unsqueeze(0).unsqueeze(1)  # linspace_x.shape = [1, 1, size]
        linspace_x = linspace_x.expand(mu.size(0), 1, size).to(x.device)   # linspace_x.shape = [배치 크기, 1, size]

        # mu, gamma, sigma를 확장하여 [배치 크기, 1, 1] 형태로 만듭니다.
        mu = mu.unsqueeze(1).unsqueeze(2)  # mu.shape = [배치 크기, 1, 1]
        gamma = gamma.unsqueeze(1).unsqueeze(2)  # gamma.shape = [배치 크기, 1, 1]
        sigma = sigma.unsqueeze(1).unsqueeze(2)  # sigma.shape = [배치 크기, 1, 1]
        amp = amp.unsqueeze(1).unsqueeze(2)  # sigma.shape = [배치 크기, 1, 1]

        # 가우시안과 로렌치안 분포 정의
        gauss = dist.Normal(mu, sigma)
        lorentz = dist.Cauchy(mu, gamma)

        # Voigt 프로파일 계산 (가우시안과 로렌치안의 곱)
        voigt = gauss.log_prob(linspace_x).exp() * lorentz.log_prob(linspace_x).exp()*amp
        #print("Voigt shape:", voigt.shape)  # 중간 결과 출력
        #print("NaN count:", torch.isnan(voigt).sum().item())  # NaN 확인
        # voigt.shape = [배치 크기, 1, size]

        return voigt
    
class ExpActivation(nn.Module):
    def __init__(self):
        super(ExpActivation, self).__init__()
    
    def forward(self, x):
        return torch.exp(x)  # 지수 활성화 함수

class HalfExpActivation(nn.Module):
    def __init__(self):
        super(HalfExpActivation, self).__init__()
        self.exp_activation = ExpActivation()
        self.relu_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 텐서를 채널 축으로 반 나누기
        split = x.size(1) // 2
        x1 = x[:, :split, :]
        x2 = x[:, split:, :]

        # 절반은 지수 활성화, 절반은 ReLU 활성화
        x1 = self.exp_activation(x1)
        x2 = self.relu_activation(x2)

        # 다시 합치기
        return torch.cat((x1, x2), dim=1)
class LogActivation(nn.Module):
    def __init__(self):
        super(LogActivation, self).__init__()
    
    def forward(self, x):
        return torch.log(x + 1e-6)  # 로그 활성화 함수 (음수 처리 X, 입력이 이미 ReLU 통과)
class HalfLogActivation(nn.Module):
    def __init__(self):
        super(HalfLogActivation, self).__init__()
        self.log_activation = LogActivation()
        self.relu_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 전체 입력에 ReLU 적용
        x = self.relu_activation(x)
        
        # 텐서를 채널 축으로 반 나누기
        split = x.size(1) // 2
        x1 = x[:, :split, :]
        x2 = x[:, split:, :]

        # 절반은 로그 활성화
        x1 = self.log_activation(x1)

        # 다시 합치기
        return torch.cat((x1, x2), dim=1)
def min_max_normalize(tensor, min_val=0.0, max_val=1.0):
    # 각 배치와 채널에 대해 최소 및 최대 값을 계산
    tensor_min = tensor.min(dim=-1, keepdim=True).values
    tensor_max = tensor.max(dim=-1, keepdim=True).values
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-10)
    normalized_tensor = normalized_tensor * (max_val - min_val) + min_val
    
    # factor 텐서 생성 (batch_size, 2, channels)
    factor = torch.cat([tensor_min, (tensor_max - tensor_min + 1e-10)], dim=-1)
    
    return normalized_tensor, factor
class SpectrumModel(nn.Module):
    def __init__(self,in_channels=1, base_channels=512):
        super(SpectrumModel, self).__init__()      
        self.encoder1 = self.conv_block(in_channels, base_channels, base_channels)
        self.encoder2 = self.conv_block(base_channels, base_channels * 2, base_channels * 2)
        self.encoder3 = self.conv_block(base_channels * 2, base_channels * 4, base_channels * 4)
        self.encoder4 = self.conv_block(base_channels * 4, base_channels * 8, base_channels * 8)
        
        self.centerconv=self.conv_block2(base_channels * 8, base_channels * 8, base_channels * 8)
        # 디코더 부분
        self.decoder4 = self.conv_block(base_channels * 16, base_channels * 4, base_channels * 8)
        self.decoder3 = self.conv_block(base_channels * 8, base_channels * 2, base_channels * 4)
        self.decoder2 = self.conv_block(base_channels * 4, base_channels, base_channels*2)
        self.decoder1 = self.conv_block(base_channels * 16, base_channels*4, base_channels*8)
        #self.decoder1 = self.conv_block(base_channels * 2, base_channels, base_channels,activation='half_log')
         # 최종 컨볼루션 레이어
        self.final_conv = nn.Conv1d(in_channels=base_channels*4, out_channels=2, kernel_size=1)
        #self.final_conv = nn.Conv1d(in_channels=base_channels, out_channels=2, kernel_size=1)
        self.relu=nn.ReLU()
        self.logact = HalfLogActivation()
    
        
        
        #self.Voigt_Conv=nn.Conv1d(in_channels=base_channels*8, out_channels=5, kernel_size=1,padding=0)
        #self.Voigt_FC1=nn.Linear(10*5,200)
        #self.Voigt_FC2=nn.Linear(200,50)
        #self.Voigt_FC3=nn.Linear(50,4)
        #self.voigt_layer = SUDOVoigtLayer()
        ## 풀링과 업샘플링
        self.downsample = nn.MaxPool1d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.upsampletw = nn.Upsample(scale_factor=4, mode='linear', align_corners=True)
        self.upsamplehex = nn.Upsample(scale_factor=8, mode='linear', align_corners=True)
        
    def conv_block(self, in_channels, out_channels, mid_channels, activation='relu'):
        if activation == 'half_log':
            activation_function = HalfLogActivation()
        if activation == 'half_exp':
            activation_function = HalfExpActivation()
        else:
            activation_function = nn.ReLU(inplace=True)
        
        block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            activation_function,
            nn.Conv1d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, padding=0),
            activation_function
        )
        return block
    def conv_block2(self, in_channels, out_channels, mid_channels, activation='relu'):
        if activation == 'half_log':
            activation_function = HalfLogActivation()
        if activation == 'half_exp':
            activation_function = HalfExpActivation()
        else:
            activation_function = nn.ReLU(inplace=True)
        
        block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            activation_function,
            nn.Conv1d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, padding=0),
        )
        return block
    def forward(self, x,mode='train'):
        # 인코더 경로
        input_size = x.shape[-1]
        x, factor = min_max_normalize(x)
        #x_np = x.cpu().numpy()  # PyTorch 텐서를 NumPy 배열로 변환
        #sg_filtered = savgol_filter(x_np, window_length=10, polyorder=5, axis=-1)
        #sg_filtered = torch.tensor(sg_filtered).to(x.device)  # NumPy 배열을 다시 PyTorch 텐서로 변환
        #difference = x - sg_filtered
        #log_transformed = torch.log1p(x)
        #x = torch.cat([x, sg_filtered, difference, log_transformed], dim=1)
        
        enc1 = self.encoder1(x)
        x = self.downsample(enc1)
        
        enc2 = self.encoder2(x)
        x = self.downsample(enc2)
        
        enc3 = self.encoder3(x)
        x = self.downsample(enc3)
        
        enc4 = self.encoder4(x)
        x = self.downsample(enc4)
        
        x=self.centerconv(x)
        #voigtC=self.Voigt_Conv(x)
        #voigtC=voigtC.view(voigtC.size(0), -1)
        #voigtC=self.relu(self.Voigt_FC1(voigtC))
        #voigtC=self.relu(self.Voigt_FC2(voigtC))
        #voigtC=self.relu(self.Voigt_FC3(voigtC))
        #voigtC=self.voigt_layer(voigtC,input_size)
        x=self.relu(x)
        # 디코더 경로
        x = self.upsample(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.decoder4(x)
        
        enc4=self.upsamplehex(enc4)
        
        x = self.upsample(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.decoder3(x)
        
        enc3 = self.upsampletw(enc3)
        
        x = self.upsample(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder2(x)
        
        enc2=self.upsample(enc2)
        
        x = self.upsample(x)
        x = torch.cat([x, enc1,enc2,enc3,enc4], dim=1)
        #x = torch.cat([x, enc1], dim=1)
        x = self.decoder1(x)

        # 최종 레이어
        x = self.final_conv(x)

        
        if mode == 'train':
            # 첫 번째 채널만 반환
            return x[:, 0, :]
        elif mode == 'train_absorbance':
            # 두 번째 채널만 반환
            return x[:, 1, :]
        elif mode == 'train_dual':
            # 두 채널 모두 반환
            return x
        elif mode == 'test_dual':
            # 두 채널 모두 반환
            x[:, 0, :]= (x[:, 0, :] * factor[:, :, 1].unsqueeze(1)) + factor[:, :, 0].unsqueeze(1)
            return x
        elif mode == 'test':
            # 첫 번째 채널에 대해 후처리 후 반환
            x = (x[:, 0, :].unsqueeze(1) * factor[:, :, 1].unsqueeze(1)) + factor[:, :, 0].unsqueeze(1)
            return x
        else:
            return 'Mode Name Error, Please Try "train", "test", "train_absorbance", or "train_dual"'

def generate_random_parameters():
    # 랜덤 파라미터 생성
    intensity = random.uniform(100, 50000)  # intensity 범위: 100 ~ 100000
    isotope = random.uniform(0, 100)         # isotope 범위: 0 ~ 100
    absorbance = random.uniform(0, 4.5)        # absorbance 범위: 0 ~ 3
    temperature = VC.Temperature(random.uniform(300, 10000), 'K')  # Tg: 300K ~ 10000K
    LH = random.uniform(10, 100)              # LH 범위: 10 ~ 50
    LC = random.uniform(0,20)               # LC 범위: 0 ~ 20
    PressureB = 0                            # PressureB는 항상 0
    shiftdiff = random.uniform(0, 0.003)     # shiftdiff 범위: 0 ~ 0.003
    noiselevel = random.uniform(10, 1000)     # noiselevel 범위: 0 ~ 1000
    X_shift = random.uniform(0.8, 0.87)
    #X_extend=random.uniform(0, 2.5)
    X_extend=2
    return intensity, isotope, absorbance, temperature, LH, LC, PressureB, shiftdiff, noiselevel,X_shift,X_extend

def set_parm(isotope=None, absorbance=None):
    # 랜덤 파라미터 생성
    intensity = 11000  # intensity 범위: 100 ~ 100000
    isotope = isotope if isotope is not None else random.uniform(0, 100)
    absorbance = absorbance if absorbance is not None else random.uniform(0, 4.5)
    temperature = VC.Temperature(3000, 'K')  # Tg: 300K ~ 10000K
    LH = 40              # LH 범위: 10 ~ 50
    LC = 10               # LC 범위: 0 ~ 20
    PressureB = 0                            # PressureB는 항상 0
    shiftdiff = 0.002    # shiftdiff 범위: 0 ~ 0.003
    noiselevel = 50     # noiselevel 범위: 0 ~ 1000
    X_shift = 0.85
    #X_extend=random.uniform(0, 2.5)
    X_extend=0.35
    return intensity, isotope, absorbance, temperature, LH, LC, PressureB, shiftdiff, noiselevel,X_shift,X_extend

def create_spectrum(solution_type='absorbance', parm="rand",isotope_in=None, absorbance_in=None):
    # 랜덤 파라미터 생성
    if parm =="rand":
        intensity, isotope, absorbance, Tg, LH, LC, PressureB, shiftdiff, noiselevel,X_shift,X_extend = generate_random_parameters()
    elif parm =="test":
        intensity, isotope, absorbance, Tg, LH, LC, PressureB, shiftdiff, noiselevel,X_shift,X_extend = set_parm(isotope_in, absorbance_in)
    # 모델 객체 생성
    spectrum_model = VC.Lithium_isotope_Object_model3(intensity=intensity,
                                                      isotope=isotope,
                                                      absorbance=absorbance,
                                                      Tg=Tg,
                                                      LH=LH+LC,
                                                      LC=LC,
                                                      PressureB=PressureB,
                                                      shiftdiff=shiftdiff)
        # X축 데이터 생성
    X = np.linspace( 669.46-X_extend, 670.544+X_extend, 160)+X_shift #667~673
    X2= np.linspace(669.46-X_extend, 670.544+X_extend, 160)+X_shift
    # 스펙트럼 데이터 생성
    Y = spectrum_model.get_intensity(X)
    noise_adder = VC.SpectralNoiseAdder(Y.copy(), add_poisson_noise=True,
                                    add_gaussian_noise=True, gaussian_noise_level=noiselevel, poisson_noise_level=1)
    Y_noisy = noise_adder.get_data()
    Y_noisy=Y_noisy-np.min(Y_noisy)
    #Y2 = spectrum_model.get_emmision_intensity(X2)
    x,factor = min_max_normalize(torch.tensor(Y_noisy))
    if solution_type=='absorbance':
        Y2 = spectrum_model.get_absorbance(X2)
    elif solution_type=='emission_normalized':   
        Y2 = spectrum_model.get_emmision_intensity(X2)/factor[1].numpy()
    elif solution_type=='emission':
        Y2 = spectrum_model.get_emmision_intensity(X2)
    elif solution_type=='origin':
        Y2 = Y.copy()
    elif solution_type == 'dual':
        # dual인 경우 emission_normalized와 absorbance 두 채널을 반환
        Y_emission_normalized = spectrum_model.get_emmision_intensity(X2) / factor[1].numpy()
        Y_absorbance = spectrum_model.get_absorbance(X2)
        Y2 = np.stack([Y_emission_normalized, Y_absorbance], axis=0)
    elif solution_type == 'dual_test':
        # dual인 경우 emission_normalized와 absorbance 두 채널을 반환
        Y_emission = spectrum_model.get_emmision_intensity(X2)
        Y_absorbance = spectrum_model.get_absorbance(X2)
        Y2 = np.stack([Y_emission, Y_absorbance], axis=0)
    # 노이즈 추가
    check=[isotope , absorbance]
    # 시각화
    #plt.figure(figsize=(10, 6))
    #plt.plot(X2, Y2, label='Target Spectrum')
    #plt.plot(X, Y_noisy, label='Noisy Spectrum', linestyle='--')
    #plt.title("Generated Spectrum with Random Parameters")
    #plt.xlabel("Wavelength")
    #plt.ylabel("Intensity")
    #plt.legend()
    #plt.show()
    return Y_noisy, Y2, check
# 데이터셋 클래스 정의
class SpectrumDataset(Dataset):
    def __init__(self,data):

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        Y_noisy, Y2 = self.data[idx]
        if np.size(Y2.shape)==1:
            return torch.tensor(Y_noisy, dtype=torch.float32).unsqueeze(0), torch.tensor(Y2, dtype=torch.float32).unsqueeze(0)
        else:
            return torch.tensor(Y_noisy, dtype=torch.float32).unsqueeze(0), torch.tensor(Y2, dtype=torch.float32)

def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'Total Parameters': total_params,
        'Trainable Parameters': trainable_params
    }