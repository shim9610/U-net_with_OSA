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

######################################################    
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
        return torch.log(x + 1e-6)  
class HalfLogActivation(nn.Module):
    def __init__(self):
        super(HalfLogActivation, self).__init__()
        self.log_activation = LogActivation()
        self.relu_activation = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu_activation(x)
        split = x.size(1) // 2
        x1 = x[:, :split, :]
        x2 = x[:, split:, :]
        x1 = self.log_activation(x1)
        return torch.cat((x1, x2), dim=1)
###############################################################
# The following custom activation functions (ExpActivation, HalfExpActivation, LogActivation, HalfLogActivation)
# were tested experimentally but did not improve model performance. Thus, these are not used in the final model.
# Included here for reference purposes only.
################################################################
    
    
def min_max_normalize(tensor, min_val=0.0, max_val=1.0):
    tensor_min = tensor.min(dim=-1, keepdim=True).values
    tensor_max = tensor.max(dim=-1, keepdim=True).values
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-10)
    normalized_tensor = normalized_tensor * (max_val - min_val) + min_val
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
        input_size = x.shape[-1]
        x, factor = min_max_normalize(x)
        enc1 = self.encoder1(x)
        x = self.downsample(enc1)
        
        enc2 = self.encoder2(x)
        x = self.downsample(enc2)
        
        enc3 = self.encoder3(x)
        x = self.downsample(enc3)
        
        enc4 = self.encoder4(x)
        x = self.downsample(enc4)
        
        x=self.centerconv(x)

        x=self.relu(x)

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

        
        if mode == 'train': #only use emission
            return x[:, 0, :]
        elif mode == 'train_absorbance':# use absorption
            return x[:, 1, :]
        elif mode == 'train_dual':# use emission&absorption
            return x
        elif mode == 'test_dual':
            x[:, 0, :]= (x[:, 0, :] * factor[:, :, 1]) + factor[:, :, 0]
            return x
        elif mode == 'test':
            x = (x[:, 0, :].unsqueeze(1) * factor[:, :, 1].unsqueeze(1)) + factor[:, :, 0].unsqueeze(1)
            return x
        else:
            return 'Mode Name Error, Please Try "train", "test", "train_absorbance", or "train_dual"'

def generate_random_parameters():
    # Random Parameter set function
    intensity = random.uniform(100, 50000)  
    isotope = random.uniform(0, 100)        
    absorbance = random.uniform(0, 4.5)     
    temperature = VC.Temperature(random.uniform(300, 10000), 'K')  
    LH = random.uniform(10, 100)              
    LC = random.uniform(0,20)               
    PressureB = 0                            
    shiftdiff = random.uniform(0, 0.003)     
    noiselevel = random.uniform(10, 1000)     
    X_shift = random.uniform(0.8, 0.87)
    #X_extend=random.uniform(0, 2.5)
    X_extend=2
    return intensity, isotope, absorbance, temperature, LH, LC, PressureB, shiftdiff, noiselevel,X_shift,X_extend

def set_parm(isotope=None, absorbance=None):
    #Custom Pram generation
    intensity = 11000  
    isotope = isotope if isotope is not None else random.uniform(0, 100)
    absorbance = absorbance if absorbance is not None else random.uniform(0, 4.5)
    temperature = VC.Temperature(3000, 'K')  
    LH = 40             
    LC = 10              
    PressureB = 0                           
    shiftdiff = 0.002   
    noiselevel = 50     
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
    spectrum_model = VC.Lithium_isotope_Object_model_main(intensity=intensity,
                                                      isotope=isotope,
                                                      absorbance=absorbance,
                                                      Tg=Tg,
                                                      LH=LH+LC,
                                                      LC=LC,
                                                      PressureB=PressureB,
                                                      shiftdiff=shiftdiff)
        # X axsis 
    X = np.linspace( 669.46-X_extend, 670.544+X_extend, 160)+X_shift #
    X2= np.linspace(669.46-X_extend, 670.544+X_extend, 160)+X_shift # for super upsampling
    # generate spectrum
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
        Y_emission_normalized = spectrum_model.get_emmision_intensity(X2) / factor[1].numpy()
        Y_absorbance = spectrum_model.get_absorbance(X2)
        Y2 = np.stack([Y_emission_normalized, Y_absorbance], axis=0)
    elif solution_type == 'dual_test':
        Y_emission = spectrum_model.get_emmision_intensity(X2)
        Y_absorbance = spectrum_model.get_absorbance(X2)
        Y2 = np.stack([Y_emission, Y_absorbance], axis=0)
    check=[isotope , absorbance]
    # debug plot
    #plt.figure(figsize=(10, 6))
    #plt.plot(X2, Y2, label='Target Spectrum')
    #plt.plot(X, Y_noisy, label='Noisy Spectrum', linestyle='--')
    #plt.title("Generated Spectrum with Random Parameters")
    #plt.xlabel("Wavelength")
    #plt.ylabel("Intensity")
    #plt.legend()
    #plt.show()
    return Y_noisy, Y2, check
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