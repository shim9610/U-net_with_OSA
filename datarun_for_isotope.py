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
    Resize the input data
    """
    original_indices = np.arange(len(data))

    resampled_indices = np.linspace(0, len(data) - 1, num_samples)

    interpolator = interpolate.interp1d(original_indices, data, kind='linear')

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

    double_array = []

    string_array = []


    pattern = re.compile(r'(\d+)\.asc$')


    for filename in os.listdir(folder_path):

        if filename.endswith('.asc'):

            match = pattern.search(filename)
            if match:

                double_array.append(float(match.group(1)))

                string_array.append(filename[:-4])  # 

    sorted_pairs = sorted(zip(double_array, string_array))
    double_array, string_array = zip(*sorted_pairs)

    double_array = list(double_array)
    string_array = list(string_array)

    return double_array, string_array
def load_asc_file_to_numpy(folder_path, file_number):

    file_path = os.path.join(folder_path, f"{str(file_number)}.asc")
    
    try:

        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                if not re.match(r'^[\d\s\.\-eE]+$', line):
                    raise ValueError(f"File {file_path} contains non-numeric data.")

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
    Exclude outliers based on a modified percentile-based IQR method.
    
    Procedure:
    1. Compute lower and upper percentiles instead of standard quartiles (25% and 75%).
    2. Define the inter-percentile range (IQR) based on these percentiles.
    3. Exclude data outside the extended bounds (1.5 × IQR) to remove extreme outliers.
    4. Calculate mean and standard deviation from the filtered data.
    
    This approach is chosen to robustly estimate mean and standard deviation
    by reducing sensitivity to extreme values in skewed or noisy data.
    """
    
    Q1 = np.percentile(data, lower_percentile)
    Q3 = np.percentile(data, upper_percentile)
    
    IQR = Q3 - Q1
    

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    

    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    

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

def linear_filtter_DC(inputdata,filename,size=100,rangeX=[187,549],bkp=0.05):
    inputdata=inputdata[rangeX[0]:rangeX[1],:]
    outputdata=np.zeros([np.size(inputdata,0),size+1])
    Xdata = inputdata[:, 0]
    outputdata[:,0]=Xdata
    for i in range(size):
        Ydata = inputdata[:, i+1]
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
        Xdata = inputdata[:, 0]
        Ydata = inputdata[:, i+1]
        # DC offset remove
        input_data = Ydata
        
        # Savitzky-Golay filtter
        #input_data = signal.savgol_filter(input_data, smoothing_window, poly_order)

        x_interp = np.linspace(668.0, 672.6, 30000)
        y_interp = np.interp(x_interp, Xdata, input_data)
        y_interp = np.clip(y_interp, a_min=0, a_max=None)
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
model = SpectrumModel()
model.eval()
weights = torch.load('logUnet_best_model_3333k_liquid512_n_weights_only.pth',
                     map_location=torch.device('cpu'))
model.load_state_dict(weights)
#big_model = torch.load('logUnet_best_model_3333k_liquid512_n.pth',
#                       map_location=torch.device('cpu'),weights_only=False )
#model_weights = big_model.state_dict()
#torch.save(model_weights, 'logUnet_best_model_3333k_liquid512_n_weights_only.pth')



folder_path = './data'
double_array, string_array = get_double_and_string_arrays_from_asc_files(folder_path)
print(double_array)
indexrange=range(0,11)
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
Tcoefficients_dip=[]
Tfiltered_data_dip=[]
TmeanV_dip=[]
TstdV_dip=[]
isotope_ratios=[]
for i in indexrange:
    name.append(double_array[i])
    isotope_ratios.append(0.95*(float(double_array[i]))+0.01*(100-float(double_array[i])))
    numpy_array = load_asc_file_to_numpy(folder_path, string_array[i])
    numpy_array=linear_filtter_DC(numpy_array,f"{str(string_array[i])}.asc").copy()
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
    coefficients_dip=get_fitvlaue(Xdata,predicted_resultD)
    Tcoefficients.append(coefficients)
    Tcoefficients_dip.append(coefficients_dip)
    meanV, stdV, filtered_data = exclude_outliers_percentile_mean_std(coefficients[:,0],lower_percentile=20, upper_percentile=80)
    meanV_dip, stdV_dip, filtered_data_dip = exclude_outliers_percentile_mean_std(coefficients_dip[:,0],lower_percentile=20, upper_percentile=80)
    fmean, fstd, ffiltered_data = exclude_outliers_percentile_mean_std(farea,lower_percentile=20, upper_percentile=80)
    Tfmean.append(fmean)
    Tfstd.append(fstd)
    Tffiltered_data.append(ffiltered_data)
    Tfiltered_data.append(filtered_data)
    Tfiltered_data_dip.append(filtered_data_dip)
    TmeanV.append(meanV)
    TstdV.append(stdV)
    TmeanV_dip.append(meanV_dip)
    TstdV_dip.append(stdV_dip)
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
TmeanV_dip=np.array(TmeanV_dip)
TstdV_dip=np.array(TstdV_dip)
isotope_ratios=np.array(isotope_ratios)
print(f"Lithium Isotope Abundance{isotope_ratios}")
print(f"Standard Daviation of Center Wavelength{TstdV}")
print(f"Center Average Wavelength{TmeanV}")
np.savetxt("Lithium Isotope Abundance.txt",isotope_ratios, delimiter='\t', fmt='%f')
np.savetxt( "Standard Daviation of Center Wavelength.txt",TstdV, delimiter='\t', fmt='%.20f')
np.savetxt( "Center Average Wavelength.txt",TmeanV, delimiter='\t', fmt='%.20f')
np.savetxt( "Standard Daviation of Center Wavelength form Absorption.txt",TstdV_dip, delimiter='\t', fmt='%.20f')
np.savetxt( "Center Average Wavelength form Absorption.txt",TmeanV_dip, delimiter='\t', fmt='%.20f')
