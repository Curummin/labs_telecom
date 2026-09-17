import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal as sig

OUT_DIR = "saida_lab3"
import os
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# PARTE 1 - MULTIPLEXACAO (FDM)
# ============================================================

def parte1_multiplexacao_fdm():
    Am, fm = 1.0, 80.0        # m(t)
    Ac, fc1 = 2.0, 1000.0     # portadora do sinal 1 (AM-DSB)
    fc2 = 2000.0              # portadora do sinal 2 (AM-DSB-SC), Ac igual

    fs = 200_000.0            # fs = 100 * fc2
    T = 0.05                  # 50 ms de sinal (5 ciclos da mensagem de 80 Hz)
    t = np.arange(0, T, 1 / fs)

    m = Am * np.cos(2 * np.pi * fm * t)                    # mensagem
    s1 = (Ac + m) * np.cos(2 * np.pi * fc1 * t)             # AM-DSB
    s2 = m * np.cos(2 * np.pi * fc2 * t)                    # AM-DSB-SC

    s_fdm = s1 + s2                                         # multiplexacao FDM

    # --- espectros (FFT) ---
    def espectro(x, fs):
        N = len(x)
        X = np.fft.rfft(x) / N
        f = np.fft.rfftfreq(N, d=1 / fs)
        return f, np.abs(X)

    f1, S1 = espectro(s1, fs)
    f2, S2 = espectro(s2, fs)
    fF, SF = espectro(s_fdm, fs)

    # --- plots: dominio do tempo ---
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axs[0].plot(t * 1e3, s1, color="tab:blue")
    axs[0].set_title("Sinal 1 - AM-DSB (fc = 1 kHz)")
    axs[1].plot(t * 1e3, s2, color="tab:orange")
    axs[1].set_title("Sinal 2 - AM-DSB-SC (fc = 2 kHz)")
    axs[2].plot(t * 1e3, s_fdm, color="tab:green")
    axs[2].set_title("Sinal multiplexado (FDM) = s1(t) + s2(t)")
    axs[2].set_xlabel("tempo (ms)")
    for ax in axs:
        ax.set_ylabel("amplitude (V)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/1_fdm_tempo.png", dpi=140)
    plt.close(fig)

    # --- plots: dominio da frequencia ---
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axs[0].plot(f1 / 1e3, S1, color="tab:blue")
    axs[0].set_title("Espectro - Sinal 1 (AM-DSB)")
    axs[1].plot(f2 / 1e3, S2, color="tab:orange")
    axs[1].set_title("Espectro - Sinal 2 (AM-DSB-SC)")
    axs[2].plot(fF / 1e3, SF, color="tab:green")
    axs[2].set_title("Espectro - Sinal multiplexado (FDM)")
    axs[2].set_xlabel("frequencia (kHz)")
    axs[2].set_xlim(0, 3)
    for ax in axs:
        ax.set_ylabel("|X(f)|")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/1_fdm_frequencia.png", dpi=140)
    plt.close(fig)

    print("[Parte 1] Graficos salvos em:",
          f"{OUT_DIR}/1_fdm_tempo.png e {OUT_DIR}/1_fdm_frequencia.png")
    print("  -> No espectro do sinal multiplexado deve aparecer:")
    print("     - raia em 1 kHz (portadora do AM-DSB) com bandas laterais em 920/1080 Hz")
    print("     - bandas laterais em 1920/2080 Hz (AM-DSB-SC, sem raia em 2 kHz)")

    return t, s1, s2, s_fdm, fs

if __name__ == "__main__":
    print("=" * 60)
    print("PARTE 1 - Multiplexacao FDM")
    print("=" * 60)
    parte1_multiplexacao_fdm()