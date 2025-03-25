import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from model_utils import SpectrumModel, SpectrumDataset, create_spectrum, get_model_size, min_max_normalize
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy import interpolate
from scipy import signal
torch.serialization.add_safe_globals([SpectrumModel])
def resampleF(data, num_samples):
    """
    데이터의 왜곡 없이 원하는 개수로 균등하게 리샘플링하는 함수.
    
    Parameters:
    data (np.ndarray): 리샘플링할 데이터 (1차원 배열).
    num_samples (int): 리샘플링 후 데이터의 샘플 수.

    Returns:
    np.ndarray: 리샘플링된 데이터 (1차원 배열).
    """
    # 원본 데이터의 인덱스 범위를 생성
    original_indices = np.arange(len(data))

    # 리샘플링된 인덱스 범위를 생성 (균등 간격)
    resampled_indices = np.linspace(0, len(data) - 1, num_samples)

    # 1차원 선형 보간기를 생성
    interpolator = interpolate.interp1d(original_indices, data, kind='linear')

    # 새로운 인덱스에 맞춰 데이터를 리샘플링
    resampled_data = interpolator(resampled_indices)
    
    return resampled_data
def my_voigt_func(x_wl, x0_wl, wG_sigma,wL_gamma,A):
    #calculate in tera Hz domain
    #input domain wavelength[nm], boradending frequency[GHz]
    c=299792458
    x=c/(x_wl*1e3)
    x0=c/(x0_wl*1e3)
    wG_sigma=wG_sigma/1e3
    wL_gamma=wL_gamma/1e3
    voigt = voigt_profile(x-x0, wG_sigma, wL_gamma)
    voigt = A*voigt/max(voigt)
    return voigt

def voigt_fit(x,y, boundsd=([669.8, 0, 0,0], [670.1, 100, 100,80000])):
    p0d=[669.92, 50,50 ,np.max(y)*1.2]
    try:
            popt, pcov =curve_fit(my_voigt_func, x,y, p0=p0d, bounds=boundsd)
    except RuntimeError as e:
        if "Optimal parameters not found" in str(e):
            print("ignore")
            popt=[0, -100, -100, -100]
        else:
            raise e
    return popt


def get_double_and_string_arrays_from_asc_files(folder_path):
    # double 값을 저장할 빈 리스트 초기화
    double_array = []
    # 파일명을 저장할 빈 리스트 초기화
    string_array = []

    # 숫자 부분을 추출하기 위한 정규 표현식 패턴
    pattern = re.compile(r'(\d+)\.asc$')

    # 폴더 내의 모든 파일에 대해 반복
    for filename in os.listdir(folder_path):
        # 파일명이 .asc로 끝나는지 확인
        if filename.endswith('.asc'):
            # 파일명에서 숫자 부분 추출
            match = pattern.search(filename)
            if match:
                # 숫자 부분을 double(float) 타입으로 변환하여 배열에 추가
                double_array.append(float(match.group(1)))
                # 확장자를 제외한 파일명을 문자열 배열에 추가
                string_array.append(filename[:-4])  # 확장자를 제외한 파일명 추가
                
    # double_array와 string_array를 함께 정렬
    sorted_pairs = sorted(zip(double_array, string_array))
    double_array, string_array = zip(*sorted_pairs)
    
    # 튜플을 리스트로 변환
    double_array = list(double_array)
    string_array = list(string_array)

    return double_array, string_array
def load_asc_file_to_numpy(folder_path, file_number):
    # 파일 경로 생성
    file_path = os.path.join(folder_path, f"{str(file_number)}.asc")
    
    try:
        # 파일 열기
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            # 문자열이 포함된 라인이 있는지 확인
            for line in lines:
                if not re.match(r'^[\d\s\.\-eE]+$', line):
                    raise ValueError(f"File {file_path} contains non-numeric data.")
            
            # 파일을 넘파이 배열로 변환
            data = np.loadtxt(file_path)
            
            return data
    
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except ValueError as ve:
        print(ve)
def sort_format(inputdata,index=range(0,101),percent=0.05,modelrange=[100,-100]):
    data=[];
    for i in index:
        Xdata=inputdata[modelrange[0]:modelrange[1],0]
        Ydata=inputdata[modelrange[0]:modelrange[1],i]
        input_data = Ydata
        #-np.min(inputdata[:,i])
        input = resampleF(input_data, 160)
        sol=np.zeros([1])
        data.append((torch.tensor(input), torch.tensor(sol)))
    return data
def get_X_data(inputdata,modelrange=[100,-100]):
    Xdata=inputdata[modelrange[0]:modelrange[1],0]
    result = resampleF(Xdata, 160)
    return result
def get_fitvlaue(Xdata,inputdata,len=100):
    coefficients=[]
    for i in range(len):
        ydata=inputdata[i+1,:,:].detach().squeeze().numpy()
        ydata=ydata-np.min(ydata)
        coef=voigt_fit(Xdata,ydata)
        coefficients.append(coef)
    return np.array(coefficients)
def exclude_outliers_percentile_mean_std(data, lower_percentile=20, upper_percentile=80):
    """
    주어진 1차원 배열에서 입력된 하위 및 상위 백분위수를 기준으로 이상치를 제외한 평균과 표준 편차를 반환하는 함수

    Args:
    data (numpy array): 1차원 배열 데이터
    lower_percentile (float): 하위 백분위수 (기본값 25)
    upper_percentile (float): 상위 백분위수 (기본값 75)

    Returns:
    mean (float): 이상치를 제외한 평균값
    std (float): 이상치를 제외한 표준 편차
    """
    
    # 하위 백분위수(Q1)와 상위 백분위수(Q3) 계산
    Q1 = np.percentile(data, lower_percentile)
    Q3 = np.percentile(data, upper_percentile)
    
    # IQR 계산
    IQR = Q3 - Q1
    
    # 이상치 기준 설정 (1.5 * IQR을 벗어난 값은 이상치로 간주)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 이상치를 제외한 데이터 필터링
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    # 평균과 표준 편차 계산
    mean_value = np.mean(filtered_data)
    std_value = np.std(filtered_data)
    
    return mean_value, std_value ,filtered_data
def get_area(Xdata,inputdata,length=100,mode="wavelength"):
    x_interp=np.linspace(668.0,672.6,30000)
    if mode == "wavelength":
        x_inte=x_interp
    elif mode == "frequency":
        c=299792458
        x_inte=c/x_interp
        
        
        
    area=np.zeros([length])
    maxV=np.zeros([length])
    for i in range(length):  
        ydata=inputdata[i+1,:,:].detach().squeeze().numpy()
        maxV[i]=np.max(ydata)
        y_interp=np.interp(x_interp,Xdata,ydata)
        y_interp = np.clip(y_interp, a_min=0, a_max=None)
        '''
        plt.clf()
        plt.plot(Xdata,ydata)
        plt.plot(x_interp,y_interp)
        plt.show()
        plt.pause(0.2) 
        '''
        area[i]=abs(integrate.simpson(y_interp,x=x_inte))
    meanV=np.mean(area)
    meanmaxV=np.mean(maxV)
    stdV=np.std(area)
    stdmax=np.std(maxV)
    return [area, meanV, stdV,maxV,meanmaxV,stdmax]

def linear_filtter_DC(inputdata,size=100,rangeX=[187,549],bkp=0.05):#paper 255,455  [120,610], [432,591]
    inputdata=inputdata[rangeX[0]:rangeX[1],:]
    outputdata=np.zeros([np.size(inputdata,0),size+1])
    Xdata = inputdata[:, 0]
    outputdata[:,0]=Xdata
    for i in range(size):
        
        Ydata = inputdata[:, i+400]
        # 1차 다항식 피팅을 통한 백그라운드 추정
        coeffs = np.polyfit(Xdata, Ydata, 1)   # coeffs[0]*x + coeffs[1]
        linear_bg = np.polyval(coeffs, Xdata)
        input_data = Ydata - linear_bg
        n = len(input_data)
        bkp_percent = int(n *bkp)  # 전체 데이터 길이의 bkp%
        outputdata[:,i+1] = input_data - np.mean(input_data[:bkp_percent])
    
    return outputdata

def smothNerr(inputdata, index=range(0,100), length=101, 
              smoothing_window=7, poly_order=3):
    area = np.zeros([np.size(index)])
    maxV = np.zeros([np.size(index)])
    
    
    for j, i in enumerate(index):
        # X축과 Y축 데이터 추출
        Xdata = inputdata[:, 0]
        Ydata = inputdata[:, i+1]
        
        # 1차 다항식 피팅을 통한 백그라운드 추정
        
        # DC 오프셋 제거 (음수값 제거)
        input_data = Ydata
        
        # Savitzky-Golay 필터를 통한 스무딩
        #input_data = signal.savgol_filter(input_data, smoothing_window, poly_order)
        
        
        # 보간 및 적분
        x_interp = np.linspace(668.0, 672.6, 30000)
        y_interp = np.interp(x_interp, Xdata, input_data)
        y_interp = np.clip(y_interp, a_min=0, a_max=None)
        '''
        plt.clf()
        plt.plot(Xdata,input_data)
        plt.plot(x_interp,y_interp)
        plt.show()
        plt.pause(0.2) 
        '''
        # 면적과 최대값 계산
        c=299792458
        x_inte=c/x_interp
        maxV[j] = np.max(input_data)
        area[j] = -(integrate.simpson(y_interp, x=x_inte))        
    meanV = np.mean(area)
    meanmaxV = np.mean(maxV)
    stdV = np.std(area)
    stdmax = np.std(maxV)
    
    return [area, meanV, stdV, maxV, meanmaxV, stdmax]
    
# 예시 사용법
# 모델 로드
#model = SpectrumModel()
#model.eval()
#weights = torch.load('logUnet_best_model_3333k_liquid512_n_weights_only.pth',
#                     map_location=torch.device('cpu'))
#model.load_state_dict(weights)
big_model = torch.load('logUnet_best_model_3333k_liquid512_n.pth',
                       map_location=torch.device('cpu'),weights_only=False )
model_weights = big_model.state_dict()
torch.save(model_weights, 'logUnet_best_model_3333k_liquid512_n_weights_only.pth')
print("가중치만 추출 완료! 용량이 훨씬 작아진 .pth 파일 생성됨.")


folder_path = './data'
double_array, string_array = get_double_and_string_arrays_from_asc_files(folder_path)
print(double_array)
indexrange=range(0,12)
Tcoefficients=[]
TstdV=[]
TmeanV=[]
name=[]
OTarea=[]
OTstdA=[]
Tarea=[]
TstdA=[]
TAmaxV=[]
OTAmaxV=[]
TAmaxstd=[]
OTAmaxstd=[]
Tfiltered_data=[]
Tfarea=[]
TfstdA=[]
Tfmean=[]
Tfstd=[]
Tffiltered_data=[]
for i in indexrange:
    name.append(double_array[i])
    numpy_array = load_asc_file_to_numpy(folder_path, string_array[i])
    numpy_array=linear_filtter_DC(numpy_array).copy()
    # Calculate the indices closest to the mean
    #selected_indices, mapped_indices = map_spectra_to_average_range(numpy_array, num_to_select=100)
    # Extract only the selected spectra based on the mapped indices
    #replace=numpy_array.copy()
    #numpy_array[:,300:400] = replace[:, (selected_indices).tolist()]
    Oarea, OmeanV, OstdV,OmaxV,OmeanmaxV,Ostdmax=smothNerr(numpy_array)
    Xdata=get_X_data(numpy_array)
    print(string_array[i])
    #print(Xdata)
    data=sort_format(numpy_array)
    dataset = SpectrumDataset(data)
    dataset=DataLoader(dataset, batch_size=101, shuffle=False)
    for inputdata, target in dataset:
        with torch.no_grad():
            predicted_params = model(inputdata,mode='test_dual')
    print(predicted_params.shape)
    predicted_resultQ=predicted_params[:,1,:].unsqueeze(1).detach()
    predicted_resultD=-np.log(abs(predicted_params[:,1,:].unsqueeze(1).detach()))
    predicted_result=predicted_params[:,0,:].unsqueeze(1).detach()
    area, AmeanV, AstdV,AmaxV,AmeanmaxV,Astdmax=get_area(Xdata,predicted_result)
    farea, fAmeanV, fAstdV,fAmaxV,fAmeanmaxV,fAstdmax=get_area(Xdata,predicted_resultD,mode="frequency")
    #print(inputdata.shape)
    coefficients=get_fitvlaue(Xdata,predicted_result)
    Tcoefficients.append(coefficients)
    meanV, stdV, filtered_data = exclude_outliers_percentile_mean_std(coefficients[:,0])
    fmean, fstd, ffiltered_data = exclude_outliers_percentile_mean_std(farea,lower_percentile=20, upper_percentile=80)
    Tfmean.append(fmean)
    Tfstd.append(fstd)
    Tffiltered_data.append(ffiltered_data)
    Tfiltered_data.append(filtered_data)
    TmeanV.append(meanV)
    TstdV.append(stdV)
    OTarea.append(Oarea)
    Tarea.append(area)
    TstdA.append(AstdV)
    OTstdA.append(OstdV)
    TAmaxV.append(AmaxV)
    OTAmaxV.append(OmaxV)
    TAmaxstd.append(Astdmax)
    OTAmaxstd.append(Ostdmax)
    Tfarea.append(farea)
    TfstdA.append(fAstdV)
    
max_length = max(len(arr) for arr in Tfiltered_data)
totaldata = np.array([np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan) for arr in Tfiltered_data]).T
max_length = max(len(arr) for arr in Tffiltered_data)
tfotaldata = np.array([np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan) for arr in Tffiltered_data]).T
totaldataA = np.array([array[:, 0].T for array in Tcoefficients]).T
X =Xdata+0.8
TmeanV=np.array(TmeanV)
TstdV=np.array(TstdV)
double_array=np.array(double_array)

for i in range(60):
    plt.clf()
    plt.rcParams.update({'font.size': 34})
    plt.plot(X, predicted_result[i,:,:].squeeze().numpy()-np.min(predicted_result[i,:,:].squeeze().numpy()), label='Predicted absorption')
    plt.plot(X, inputdata[i,:,:].squeeze().numpy(), label='Real Absorption')
    plt.legend(fontsize=15)
    plt.show()
    plt.pause(0.2) 
#286:446
