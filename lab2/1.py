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

# ==========================================
# Parâmetros Iniciais
# ==========================================
fc = 1000
fs = fc * 100                   
t = np.arange(0, 0.02, 1/fs)    

# Modulante
Em = 8
fm_freq = 200
modulante_norm = np.cos(2 * np.pi * fm_freq * t) 
modulante = Em * np.cos(2 * np.pi * fm_freq * t)

# Portadora
portadora = np.cos(2 * np.pi * fc * t)

# ==========================================
# Exercício 1 (Tempo)
# ==========================================

# ------AM_DSB------
ma = 0.5
A = Em / ma
s_am_dsb = am_dsb(Em=Em, modulante=modulante_norm, portadora=portadora, A=A, ma=ma)
envoltoria_superior = A * (1 + ma * modulante_norm)
envoltoria_inferior = -A * (1 + ma * modulante_norm)

# ------PM------
Kp = np.pi / 16     
s_pm = pm(modulante, 1, Kp, fc, t)

# ------FM------
Kf = 50 
s_fm = fm(1, fc, Kf, Em, fm_freq, t)

# Gráfico AM-DSB
plt.figure(figsize=(10, 4))
plt.plot(t, s_am_dsb, label='Sinal AM-DSB', color='blue')
plt.plot(t, envoltoria_superior, '--', label='Envoltória Superior', color='red', linewidth=2)
plt.plot(t, envoltoria_inferior,  '--', label='Envoltória Inferior', color='red', linewidth=2)
plt.title('Domínio do Tempo: Sinal AM-DSB')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Gráficos Modulação Angular
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True) # Correção: 'pfig' alterado para 'fig'

axs[0].plot(t, modulante, color='orange')
axs[0].set_title('Sinal Modulante')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

axs[1].plot(t, s_pm, color='green')
axs[1].set_title('Domínio do Tempo: Sinal PM')
axs[1].set_ylabel('Amplitude')
axs[1].grid(True)

axs[2].plot(t, s_fm, color='purple')
axs[2].set_title('Domínio do Tempo: Sinal FM')
axs[2].set_xlabel('Tempo (s)')
axs[2].set_ylabel('Amplitude')
axs[2].grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# Exercício 2 (Frequência)
# ==========================================

s_am_dsb_sc = am_dsb_sc(modulante, portadora)
s_am_ssb = (modulante * portadora) - (Em * np.sin(2 * np.pi * fm_freq * t) * np.sin(2 * np.pi * fc * t)) 

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

plot_fft_individual(s_am_dsb_sc, 'Domínio da Frequência: AM-DSB-SC')
plot_fft_individual(s_am_dsb, 'Domínio da Frequência: AM-DSB')
plot_fft_individual(s_am_ssb, 'Domínio da Frequência: AM-SSB (USB)')
plot_fft_individual(s_fm, 'Domínio da Frequência: FM')