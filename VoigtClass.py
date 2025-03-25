# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:00:01 2022

@author: shim9610
#import os
#import matplotlib.pyplot as plt
#import pandas as pd
#from scipy.optimize import curve_fit
#import xml.etree.ElementTree as ET
#import re
"""
#%% import module

import numpy as np
import math
from scipy import special
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
from scipy.signal import correlate
#import glob
import os
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
#%% calculate Maxwellian distribution for calculate rate constant   MDcal(mxrange,mirange,Te,dx)
'''
ref : R. K. Gangwar, JOURNAL OF APPLIED PHYSICS 111, 053307 (2012) for calculate Rate coefficient 
F(E)=E**0.5*f(E)
f(E)=Maxwellian distribution
This function is E*F(E)
Dimension jule
'''
class SpectralNoiseAdder:
    def __init__(self, intensity, add_gaussian_noise=True, add_poisson_noise=True, gaussian_noise_level=70, poisson_noise_level=1):
        """
        Initialize the SpectralNoiseAdder class.
        
        Parameters:
        - intensity: The original intensity data (Y-axis data).
        - add_gaussian_noise: Boolean flag to add Gaussian noise.
        - add_poisson_noise: Boolean flag to add Poisson noise.
        - gaussian_noise_level: Standard deviation of Gaussian noise.
        - poisson_noise_level: Scaling factor for Poisson noise.
        """
        self.intensity = intensity
        self.add_gaussian_noise_flag = add_gaussian_noise
        self.add_poisson_noise_flag = add_poisson_noise
        self.gaussian_noise_level = gaussian_noise_level
        self.poisson_noise_level = poisson_noise_level
        
    def add_gaussian_noise(self, data):
        """
        Add Gaussian noise to the data.
        
        Parameters:
        - data: The data to which Gaussian noise will be added.
        
        Returns:
        - data with Gaussian noise added.
        """
        gaussian_noise = np.random.normal(0, self.gaussian_noise_level, size=data.shape)
        return gaussian_noise
    
    def add_poisson_noise(self, data):
        """
        Add Poisson noise to the data.
        
        Parameters:
        - data: The data to which Poisson noise will be added.
        
        Returns:
        - data with Poisson noise added.
        """
        scaled_data = data / self.poisson_noise_level
        scaled_data = np.clip(scaled_data, 0, None)  # 음수 값을 0으로 조정
        poisson_noise = np.random.poisson(scaled_data).astype(float) * self.poisson_noise_level - data
        return poisson_noise
    
    def get_data(self):
        """
        Get the data with the specified noises added.
        
        Returns:
        - A dictionary with original and noisy data.
        """
        noisy_data = self.intensity.copy()
        if self.add_gaussian_noise_flag:
            noisy_data = noisy_data + self.add_gaussian_noise(self.intensity)
        if self.add_poisson_noise_flag:
            noisy_data = noisy_data + self.add_poisson_noise(self.intensity)
        return noisy_data

# 사용 예시
wavelength = np.linspace(400, 700, 3000)
intensity = norm.pdf(wavelength, loc=550, scale=20)
intensity = intensity / np.max(intensity) * 3000  # 정규화하여 피크를 3000으로 설정

noise_adder = SpectralNoiseAdder(intensity, add_gaussian_noise=True, add_poisson_noise=True, gaussian_noise_level=70, poisson_noise_level=1)
noisy_data = noise_adder.get_data()
class Temperature:
    def __init__(self, value,dimension = 'K',model='Lithium'):
        self.value=value
        self.model = model
        self.dimension = dimension
        self.mass=None
        self.get_mass_from_model()

    def convert_dimension_back(self, target_dimension):
        # Convert from current dimension to Kelvin
        if self.dimension == 'K':
            value_in_K = self.value
        elif self.dimension == 'C':
            value_in_K = self.value + 273.15
        elif self.dimension == 'F':
            value_in_K = (self.value + 459.67) * 5/9
        elif self.dimension == 'eV':
            value_in_K = self.value * 11604.525
        else:
            raise ValueError(f"Unknown dimension: {self.dimension}")
        # Convert from Kelvin to target dimension
        if target_dimension == 'K':
            return value_in_K
        elif target_dimension == 'C':
            return value_in_K - 273.15
        elif target_dimension == 'F':
            return value_in_K * 9/5 - 459.67
        elif target_dimension == 'eV':
            return value_in_K / 11604.525
        else:
            raise ValueError(f"Unknown target dimension: {target_dimension}")
    def convert_dimension(self, target_dimension):
        self.value=self.convert_dimension_back(target_dimension)
        self.dimension=target_dimension
    def calGHz(self, wavelength):
        Kelvin=self.convert_dimension_back('K')
        c = 299792458  # speed of light in m/s
        cfre = c / (wavelength * 1e-9) * 1e-9
        k = 1.3806504e-23  # Boltzmann constant in J/K
        GHz = cfre * (8 * k * Kelvin * np.log(2) / self.mass / c / c) ** 0.5
        return GHz    
    def get_mass_from_model(self):
        match self.model:
            case 'Lithium':
                self.mass = 1.1526219e-26  # kg
            case 'Lithium-6':
                self.mass = 9.9883414e-27  # kg
            case 'Lithium-7':
                self.mass = 1.1650482e-26  # kg
            case 'Hydrogen':
                self.mass = 1.6735575e-27  # kg
            case 'Deuterium':
                self.mass = 3.344493e-27  # kg
            case 'Tritium':
                self.mass = 5.007356e-27  # kg
            case 'Helium':
                self.mass = 6.6464764e-27  # kg
            case 'Argon':
                self.mass = 6.6335209e-26  # kg
            case _:
                self.mass = None  # or some default value
    def __add__(self, other):
        if isinstance(other, Temperature):
            return self.value + other.value
        else:
            return self.value + other
    def __radd__(self, other):
        return self.__add__(other)
    def __sub__(self, other):
        if isinstance(other, Temperature):
            return self.value - other.value
        else:
            return self.value - other
    def __rsub__(self, other):
        if isinstance(other, Temperature):
            return other.value - self.value
        else:
            return other - self.value
    def __mul__(self, other):
        if isinstance(other, Temperature):
            return self.value * other.value
        else:
            return self.value * other
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(self, other):
        if isinstance(other, Temperature):
            return self.value / other.value
        else:
            return self.value / other
    def __rtruediv__(self, other):
        if isinstance(other, Temperature):
            return other.value / self.value
        else:
            return other / self.value
    def __pow__(self, exponent):
        if isinstance(exponent, Temperature):
            return self.value ** exponent.value
        else:
            return self.value ** exponent
    def __rpow__(self, base):
        return base ** self.value
    def __array__(self, dtype=None):
        return np.array(self.value, dtype=dtype)
    def __str__(self):
        return f"Temperature value: {self.value} {self.dimension}, Model: {self.model}, Mass: {self.mass} kg"
    def __repr__(self):
        return f"Temperature(value={self.value}, dimension='{self.dimension}', model='{self.model}')"
class StarkObject:
    def __init__(self, Te,Ne):
        if isinstance(Te, Temperature):
            self.Te = Te.convert_dimension_back('K')  # Temperature 객체인 경우 K로 변환
        else:
            self.Te = Te  # 숫자인 경우 그대로 사용
        self.Ne=Ne
        self.model='Lithium_670.8nm'
        self.stark_ref=[]
        self.stark_width=[]
        self.stark_shift=[]
        self.stark_ref=self.calculate_stark_width()
        self.convert_peak()
        
    def calculate_stark_width(self):#https://stark-b.obspm.fr/index.php/data/ion/98
        c=float(299792458)
        a0=-1.63869
        a1=-1.70083
        a2=0.24083
        b0=3.63941
        b1=-2.06984
        b2=0.27181
        result=np.zeros([2])
        Ne=self.Ne/1e13
        result[0]=10**(a0+a1*np.log10(self.Te)+a2*np.log10(self.Te)**2)*Ne#basic=Å 
        result[1]=(b0+b1*np.log10(self.Te)+b2*np.log10(self.Te)**2)*result[0]
        result[1]=result[1]
        #lam=(result[0]*10+670.791425)*1e-9
        #result[0]=abs(c/lam-c/(670.791425*1e-9))*1e-9
        return result
    def convert_peak(self):
        c=float(299792458)
        match self.model:
            case 'Lithium_670.8nm':
                lam=670.791425*1e-9     
        dlam = lam + self.stark_ref[0]*1e-10
        self.stark_width=abs((c/lam)-(c/dlam))/1e9#[GHz]
        self.stark_shift=self.stark_ref[1]/1e10#[m]
class TransitionLine:
    def __init__(self,wavelength,intensity,species="Lithium",A=None,f=None,gf=None,gi=None,Te=None,Ne=None,Tg=None,model='Voigt',Lorentz_width=0,Gauss_width=0,instrumental_broadening=0,pressureB=0):
        self.species=species
        self.wavelength=wavelength/1e9
        self.A=A
        self.c=299792458
        self.f=f
        self.gi=gi
        self.gf=gf
        self.model=model
        self.intensity=intensity
        self.Gauss_width=0
        self.Loerntz_width=0
        self.PressureB=pressureB
        self.instrumental_broadening=abs(self.c/(wavelength-instrumental_broadening)-self.c/wavelength)
        if Tg is not None:
            if isinstance(Tg, Temperature):
                self.Tg=Tg
                self.Tg.model=species
                self.Tg.get_mass_from_model()
                self.Gauss_width=Tg.calGHz(wavelength)
            else:
                self.Tg=Temperature(Tg,model=species)
                self.Gauss_width=Tg.calGHz(wavelength)
        else:
            self.Tg=None
            self.Gauss_width=Gauss_width
        if Te and Ne is not None:
            self.stark=StarkObject(Te,Ne)
            self.Loerntz_width=self.stark.stark_width
        else:
            self.stark=None
            self.Loerntz_width=Lorentz_width
        
    def get_intensity(self,X):#X is wavelength in nm
        if self.model=='Voigt':
            #print(f'IN f Loerntz_width : {self.Loerntz_width}, Gauss_width : {self.Gauss_width}')
            if self.Loerntz_width is not None or self.Gauss_width is not None or self.instrumental_broadening is not None:
                if self.stark is not None:
                    center=self.wavelength+self.stark.stark_shift
                else:
                    center=self.wavelength
                X=X*1e-9
                frequency=self.c/X/1e9#[GHz]
                center = self.c/center/1e9
                G=self.Gauss_width/(2*np.log(2))**0.5
                #print(f'Gauss_width : {self.Gauss_width}, Lorentz_width : {self.Loerntz_width}, frequency : {frequency}, center : {center}, intensity : {self.intensity}, instrumental_broadening : {self.instrumental_broadening}')
                scale = special.voigt_profile(0, (G+self.instrumental_broadening)*0.5, (self.Loerntz_width+self.PressureB)*0.5, out=None)
                #print(f'scale : {scale}')  
                result = self.intensity*special.voigt_profile(frequency-center, G, (self.Loerntz_width+self.PressureB)*0.5, out=None)/scale
                return  result
            else:
                print('Voigt model requires both Lorentz and Gauss width')
                print(f'Lorentz : {self.Loerntz_width},Gauss : {self.Gauss_width}')
                return np.where(np.isclose(X , self.wavelength*1e9,rtol=1e-09, atol=1e-09), self.intensity, 0)
        else:
            return None
class Self_Absorption(TransitionLine):
    def __init__(self, wavelength, absorbance, species="Lithium", A=None, f=None, gf=None, gi=None, Te=None, Ne=None, Tg=None, model='Voigt', Lorentz_width=0, Gauss_width=0, instrumental_broadening=0, pressureB=0):
        super().__init__(wavelength, intensity=None, species=species, A=A, f=f, gf=gf, gi=gi, Te=Te, Ne=Ne, Tg=Tg, model=model, Lorentz_width=Lorentz_width, Gauss_width=Gauss_width, instrumental_broadening=instrumental_broadening, pressureB=pressureB)
        self.absorbance = absorbance
    def get_intensity(self,X):#X is wavelength in nm
        if self.model=='Voigt':
            #print(f'IN f Loerntz_width : {self.Loerntz_width}, Gauss_width : {self.Gauss_width}')
            if self.Loerntz_width != 0 or self.Gauss_width != 0 or self.instrumental_broadening != 0:
                if self.stark is not None:
                    center=self.wavelength+self.stark.stark_shift
                else:
                    center=self.wavelength
                X=X*1e-9
                frequency=self.c/X/1e9#[GHz]
                center = self.c/center/1e9
                G=self.Gauss_width/(2*np.log(2))**0.5
                #print(f'Gauss_width : {self.Gauss_width}, Lorentz_width : {self.Loerntz_width}, frequency : {frequency}, center : {center}, intensity : {self.intensity}, instrumental_broadening : {self.instrumental_broadening}')
                scale = special.voigt_profile(0, (G+self.instrumental_broadening)*0.5, (self.Loerntz_width+self.PressureB)*0.5, out=None)
                #print(f'scale : {scale}')  
                result = np.exp(-self.absorbance*special.voigt_profile(frequency-center, (G+self.instrumental_broadening)*0.5, (self.Loerntz_width+self.PressureB)*0.5, out=None)/scale)
                return  result
            else:
                print('Voigt model requires both Lorentz and Gauss width')
                return np.where(np.isclose(X , self.wavelength*1e9,rtol=1e-09, atol=1e-09), np.exp(-self.absorbance), 0)
        else:
            return None
class Lithium_isotope_Object:
    def __init__(self,intensity=1,isotope=7.5,absorbance=0.1,TeH=Temperature(0.5,'eV'),Ned=0.8e17,TeC=Temperature(300,'K'),Neld=0.8e10,PressureB=0):
        
        if isotope is not None:
            self.lithium6_rate=isotope/7.5
            self.lithium7_rate=(100-isotope)/92.5    
        else:
            self.lithium6_rate=[]
            self.lithium7_rate=[]
        self.intensity=intensity
        self.L7D1_i=1
        self.L6D1_i=0.081
        self.L7D2_i=0.5
        self.L6D2_i=0.041
        self.L7D1=TransitionLine(670.7760,self.L7D1_i*self.lithium7_rate,species="Lithium-7",Te=TeH,Ne=Ned,Tg=TeH,pressureB=PressureB,instrumental_broadening=0.025)
        self.L6D1=TransitionLine(670.7918,self.L6D1_i*self.lithium6_rate,species="Lithium-6",Te=TeH,Ne=Ned,Tg=TeH,pressureB=PressureB,instrumental_broadening=0.025)
        self.L7D2=TransitionLine(670.7911,self.L7D2_i*self.lithium7_rate,species="Lithium-7",Te=TeH,Ne=Ned,Tg=TeH,pressureB=PressureB,instrumental_broadening=0.025)
        self.L6D2=TransitionLine(670.8068,self.L6D2_i*self.lithium6_rate,species="Lithium-6",Te=TeH,Ne=Ned,Tg=TeH,pressureB=PressureB,instrumental_broadening=0.025)
        self.L7D1self=Self_Absorption(670.7760,absorbance*self.L7D1_i*self.lithium7_rate,species="Lithium-7",Te=TeC,Ne=Neld,Tg=TeC,pressureB=PressureB)
        self.L6D1self=Self_Absorption(670.7918,absorbance*self.L6D1_i*self.lithium6_rate,species="Lithium-6",Te=TeC,Ne=Neld,Tg=TeC,pressureB=PressureB)
        self.L7D2self=Self_Absorption(670.7911,absorbance*self.L7D2_i*self.lithium7_rate,species="Lithium-7",Te=TeC,Ne=Neld,Tg=TeC,pressureB=PressureB)
        self.L6D2self=Self_Absorption(670.8068,absorbance*self.L6D2_i*self.lithium6_rate,species="Lithium-6",Te=TeC,Ne=Neld,Tg=TeC,pressureB=PressureB)
    def get_intensity(self,X):
        emmision=self.L7D1.get_intensity(X)+self.L6D1.get_intensity(X)+self.L7D2.get_intensity(X)+self.L6D2.get_intensity(X)
        emmision=emmision/np.max(emmision)*self.intensity
        selfabsorption=self.L7D1self.get_intensity(X)*self.L6D1self.get_intensity(X)*self.L7D2self.get_intensity(X)*self.L6D2self.get_intensity(X)
        return emmision*selfabsorption
    def print_shift(self):
        print(f'emissiton :{self.L7D1.stark.stark_shift}, self absorption : {self.L7D1self.stark.stark_shift}')
class Lithium_isotope_Object_model2:
    def __init__(self,intensity=1,isotope=7.5,absorbance=0.1,TgH=Temperature(300,'K'),TgC=Temperature(300,'K'),TeH=Temperature(0.5,'eV'),Ned=0.8e17,TeC=Temperature(300,'K'),Neld=0.8e10,PressureB=0):
        if ~isinstance(TgC,Temperature):
            TgC=Temperature(TgC,'K')
        if ~isinstance(TgH,Temperature):
            TgH=Temperature(TgH,'K')
        if isotope is not None:
            self.lithium6_rate=isotope/7.5
            self.lithium7_rate=(100-isotope)/92.5    
        else:
            self.lithium6_rate=[]
            self.lithium7_rate=[]
        self.intensity=intensity
        self.absorbance=absorbance
        self.L7D1_i=1
        self.L6D1_i=0.081
        self.L7D2_i=0.5
        self.L6D2_i=0.041
        self.L7D1=TransitionLine(670.7760,self.L7D1_i*self.lithium7_rate,species="Lithium-7",Te=TeH,Ne=Ned,Tg=TgH,pressureB=PressureB)
        self.L6D1=TransitionLine(670.7918,self.L6D1_i*self.lithium6_rate,species="Lithium-6",Te=TeH,Ne=Ned,Tg=TgH,pressureB=PressureB)
        self.L7D2=TransitionLine(670.7911,self.L7D2_i*self.lithium7_rate,species="Lithium-7",Te=TeH,Ne=Ned,Tg=TgH,pressureB=PressureB)
        self.L6D2=TransitionLine(670.8068,self.L6D2_i*self.lithium6_rate,species="Lithium-6",Te=TeH,Ne=Ned,Tg=TgH,pressureB=PressureB)
        self.L7D1self=TransitionLine(670.7760,self.L7D1_i*self.lithium7_rate,species="Lithium-7",Te=TeC,Ne=Neld,Tg=TgC,pressureB=PressureB)
        self.L6D1self=TransitionLine(670.7918,self.L6D1_i*self.lithium6_rate,species="Lithium-6",Te=TeC,Ne=Neld,Tg=TgC,pressureB=PressureB)
        self.L7D2self=TransitionLine(670.7911,self.L7D2_i*self.lithium7_rate,species="Lithium-7",Te=TeC,Ne=Neld,Tg=TgC,pressureB=PressureB)
        self.L6D2self=TransitionLine(670.8068,self.L6D2_i*self.lithium6_rate,species="Lithium-6",Te=TeC,Ne=Neld,Tg=TgC,pressureB=PressureB)
    def get_intensity(self,X):
        emmision=self.L7D1.get_intensity(X)+self.L6D1.get_intensity(X)+self.L7D2.get_intensity(X)+self.L6D2.get_intensity(X)
        emmision=emmision/np.max(emmision)*self.intensity
        selfabsorption=self.L7D1self.get_intensity(X)+self.L6D1self.get_intensity(X)+self.L7D2self.get_intensity(X)+self.L6D2self.get_intensity(X)
        selfabsorption=np.exp(-selfabsorption/np.max(selfabsorption)*self.absorbance)
        return emmision*selfabsorption
    def print_shift(self):
        print(f'emissiton :{self.L7D1.stark.stark_shift}, self absorption : {self.L7D1self.stark.stark_shift}')
    def get_emmision_intensity(self,X):
        emmision=self.L7D1.get_intensity(X)+self.L6D1.get_intensity(X)+self.L7D2.get_intensity(X)+self.L6D2.get_intensity(X)
        emmision=emmision/np.max(emmision)*self.intensity
        return emmision
    def apply_gaussian_broadening(self, X, spectrum, resolution_nm=0.024):
        c = 299792.458  # Speed of light in km/s
        center_wavelength_nm = np.mean(X)
        center_frequency_GHz = c / center_wavelength_nm  # GHz
        resolution_GHz = c / (center_wavelength_nm - resolution_nm) - center_frequency_GHz

        # Convert FWHM to standard deviation
        sigma_GHz = resolution_GHz / (2 * np.sqrt(2 * np.log(2)))  # GHz

        # Calculate the frequency difference per pixel in GHz
        frequency_diff_per_pixel = abs(np.mean(np.diff(c / X)))  # Convert X from nm to GHz and take the mean difference

        # Convert sigma from GHz to pixels
        sigma_pixels = sigma_GHz / frequency_diff_per_pixel

        broadened_spectrum = gaussian_filter1d(spectrum, sigma_pixels)
        return broadened_spectrum
    def get_broadened_intensity_emmision(self, X):
        spectrum = self.get_emmision_intensity(X)
        broadened_spectrum = self.apply_gaussian_broadening(X, spectrum)
        return broadened_spectrum
  
    def get_broadened_intensity(self, X):
        spectrum = self.get_intensity(X)
        broadened_spectrum = self.apply_gaussian_broadening(X, spectrum)
        return broadened_spectrum
class Lithium_isotope_Object_model3:
    def __init__(self,intensity=1,isotope=7.5,absorbance=2.5,Tg=Temperature(300,'K'),LH=0,LC=0 ,PressureB=0, shiftdiff=0,instrumental_broadening=0):
        if ~isinstance(Tg,Temperature):
            TgC=Temperature(Tg,'K')
            TgH=Temperature(Tg,'K')
        else:
            TgC=Tg
            TgH=Tg
        if isotope is not None:
            self.lithium6_rate=isotope/7.5
            self.lithium7_rate=(100-isotope)/92.5    
        else:
            self.lithium6_rate=[]
            self.lithium7_rate=[]
        self.intensity=intensity
        self.absorbance=absorbance
        self.L7D1_i=1
        self.L6D1_i=0.081
        self.L7D2_i=0.5
        self.L6D2_i=0.041
        self.L7D1=TransitionLine(670.7760,self.L7D1_i*self.lithium7_rate,species="Lithium-7",Lorentz_width= LH,Tg=TgH,pressureB=PressureB)
        self.L6D1=TransitionLine(670.7918,self.L6D1_i*self.lithium6_rate,species="Lithium-6",Lorentz_width= LH,Tg=TgH,pressureB=PressureB)
        self.L7D2=TransitionLine(670.7911,self.L7D2_i*self.lithium7_rate,species="Lithium-7",Lorentz_width= LH,Tg=TgH,pressureB=PressureB)
        self.L6D2=TransitionLine(670.8068,self.L6D2_i*self.lithium6_rate,species="Lithium-6",Lorentz_width= LH,Tg=TgH,pressureB=PressureB)
        self.L7D1self=TransitionLine(670.7760+shiftdiff,self.L7D1_i*self.lithium7_rate,species="Lithium-7",Lorentz_width= LC,Tg=TgC,pressureB=PressureB)
        self.L6D1self=TransitionLine(670.7918+shiftdiff,self.L6D1_i*self.lithium6_rate,species="Lithium-6",Lorentz_width= LC,Tg=TgC,pressureB=PressureB)
        self.L7D2self=TransitionLine(670.7911+shiftdiff,self.L7D2_i*self.lithium7_rate,species="Lithium-7",Lorentz_width= LC,Tg=TgC,pressureB=PressureB)
        self.L6D2self=TransitionLine(670.8068+shiftdiff,self.L6D2_i*self.lithium6_rate,species="Lithium-6",Lorentz_width= LC,Tg=TgC,pressureB=PressureB)
    def get_intensity(self,X):
        emmision=self.L7D1.get_intensity(X)+self.L6D1.get_intensity(X)+self.L7D2.get_intensity(X)+self.L6D2.get_intensity(X)
        emmision=emmision/np.max(emmision)*self.intensity
        selfabsorption=self.L7D1self.get_intensity(X)+self.L6D1self.get_intensity(X)+self.L7D2self.get_intensity(X)+self.L6D2self.get_intensity(X)
        selfabsorption=np.exp(-selfabsorption/np.max(selfabsorption)*self.absorbance)
        return emmision*selfabsorption
    def print_shift(self):
        print(f'emissiton :{self.L7D1.stark.stark_shift}, self absorption : {self.L7D1self.stark.stark_shift}')
    def get_emmision_intensity(self,X):
        emmision=self.L7D1.get_intensity(X)+self.L6D1.get_intensity(X)+self.L7D2.get_intensity(X)+self.L6D2.get_intensity(X)
        emmision=emmision/np.max(emmision)*self.intensity
        return emmision
    def get_absorbance(self,X):
        selfabsorption=self.L7D1self.get_intensity(X)+self.L6D1self.get_intensity(X)+self.L7D2self.get_intensity(X)+self.L6D2self.get_intensity(X)
        selfabsorption=np.exp(-selfabsorption/np.max(selfabsorption)*self.absorbance)
        return selfabsorption
    def apply_gaussian_broadening(self, X, spectrum, resolution_nm=0.024):
        c = 299792.458  # Speed of light in km/s
        center_wavelength_nm = np.mean(X)
        center_frequency_GHz = c / center_wavelength_nm  # GHz
        resolution_GHz = c / (center_wavelength_nm - resolution_nm) - center_frequency_GHz

        # Convert FWHM to standard deviation
        sigma_GHz = resolution_GHz / (2 * np.sqrt(2 * np.log(2)))  # GHz

        # Calculate the frequency difference per pixel in GHz
        frequency_diff_per_pixel = abs(np.mean(np.diff(c / X)))  # Convert X from nm to GHz and take the mean difference

        # Convert sigma from GHz to pixels
        sigma_pixels = sigma_GHz / frequency_diff_per_pixel

        broadened_spectrum = gaussian_filter1d(spectrum, sigma_pixels)
        return broadened_spectrum
    def get_broadened_intensity_emmision(self, X,resolution_nm=0.024):
        spectrum = self.get_emmision_intensity(X)
        broadened_spectrum = self.apply_gaussian_broadening(X, spectrum,resolution_nm=resolution_nm)
        return broadened_spectrum
  
    def get_broadened_intensity(self, X,resolution_nm=0.024):
        spectrum = self.get_intensity(X)
        broadened_spectrum = self.apply_gaussian_broadening(X, spectrum,resolution_nm=resolution_nm)
        return broadened_spectrum    
    

class Lithium_isotope_Object_preFit:
    def __init__(self,intensity=1,isotope=7.5,absorbance=0.1,TgH=Temperature(300,'K'),TgC=Temperature(300,'K'),TeH=Temperature(0.5,'eV'),Ned=0.8e17,PressureB=0):
        if ~isinstance(TgC,Temperature):
            TgC=Temperature(TgC,'K')
        if ~isinstance(TgH,Temperature):
            TgH=Temperature(TgH,'K')
        if isotope is not None:
            self.lithium6_rate=isotope/7.5
            self.lithium7_rate=(100-isotope)/92.5    
        else:
            self.lithium6_rate=[]
            self.lithium7_rate=[]
        self.intensity=intensity
        self.absorbance=absorbance
        self.L7D1_i=1
        self.L6D1_i=0.081
        self.L7D2_i=0.5
        self.L6D2_i=0.041
        self.L7D1=TransitionLine(670.7760,self.L7D1_i*self.lithium7_rate,species="Lithium-7",Te=TeH,Ne=Ned,Tg=TgH,pressureB=PressureB)
        self.L6D1=TransitionLine(670.7918,self.L6D1_i*self.lithium6_rate,species="Lithium-6",Te=TeH,Ne=Ned,Tg=TgH,pressureB=PressureB)
        self.L7D2=TransitionLine(670.7911,self.L7D2_i*self.lithium7_rate,species="Lithium-7",Te=TeH,Ne=Ned,Tg=TgH,pressureB=PressureB)
        self.L6D2=TransitionLine(670.8068,self.L6D2_i*self.lithium6_rate,species="Lithium-6",Te=TeH,Ne=Ned,Tg=TgH,pressureB=PressureB)

    def get_intensity(self,X):
        emmision=self.L7D1.get_intensity(X)+self.L6D1.get_intensity(X)+self.L7D2.get_intensity(X)+self.L6D2.get_intensity(X)
        emmision=emmision/np.max(emmision)*self.intensity
        return emmision
    def print_shift(self):
        print(f'emissiton :{self.L7D1.stark.stark_shift}')
