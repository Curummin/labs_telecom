import numpy as np
import matplotlib.pyplot as plt

def am_dsb(Em, modulante, portadora, A=0, ma=0):
    return A * (1 + ma * modulante) * portadora

def am_dsb_sc(modulante, portadora):
    return modulante * portadora

def am_ssb(Em, modulante, fm_freq, portadora, fc, t, filtro=0):
    x = am_dsb_sc(modulante=modulante, portadora=portadora)
    if(filtro):
        return x - (Em * np.sin(2 * np.pi * fm_freq * t) * np.sin(2 * np.pi * fc * t)) # USB
    else:
        return x + (Em * np.sin(2 * np.pi * fm_freq * t) * np.sin(2 * np.pi * fc * t)) # LSB/DSB
    
def pm(modulante, A, Kp, f_portadora, t):  
    return A * np.cos(2 * np.pi * f_portadora * t + Kp * modulante)

def fm(A, f_portadora, Kf, Am, fm_freq, t):
    betha = (Kf * Am) / fm_freq
    return A * np.cos(2 * np.pi * f_portadora * t + betha * np.sin(2 * np.pi * fm_freq * t))

def plot_fft_individual(signal, title):
    N = len(signal)
    fft_val = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    
    fft_val_shifted = np.fft.fftshift(fft_val)
    freqs_shifted = np.fft.fftshift(freqs)
    
    # CORREÇÃO: Para espectros com frequências negativas e positivas (bilateral), 
    # a normalização correta é dividir apenas por N.
    fft_mag = np.abs(fft_val_shifted) / N
    
    plt.figure(figsize=(10, 4))
    plt.plot(freqs_shifted, fft_mag, color='black')
    plt.xlim(-2000, 2000) 
    plt.title(title)
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()    

fc = 1e6
fs = fc * 100

t = np.arange(0, 0.02, 1/fs) 
Kf = 5 
s_fm = fm(1, 1e6, Kf, 10 , 500, t)

