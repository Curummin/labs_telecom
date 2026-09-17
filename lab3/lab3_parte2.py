import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal as sig
import pyaudio
import wave
import keyboard
import time
import os
try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except Exception:
    SOUNDDEVICE_OK = False

OUT_DIR = "saida_lab3"
os.makedirs(OUT_DIR, exist_ok=True)

FORMAT = pyaudio.paInt16

def gravar_ou_gerar_audio(duracao=2.0, fs=16000):
    try:
        audio_obj = pyaudio.PyAudio()
        CHUNK = 1024
        
        # Usa o input nativo do Python que pausa a execução até o usuário apertar Enter
        input(f"[Parte 2a] Pressione ENTER para começar a gravar a {fs} Hz...")
        
        stream = audio_obj.open(format=FORMAT, 
                                channels=1, 
                                rate=fs, 
                                input=True, 
                                frames_per_buffer=CHUNK)
        
        frames = []
        print("[Parte 2a] Gravando... Pressione Ctrl+C para parar.")
        
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except KeyboardInterrupt:
                # O Ctrl+C aciona essa exceção, interrompendo o loop
                print("\n[Parte 2a] Parando gravação...")
                break
                
        stream.stop_stream()
        stream.close()
        audio_obj.terminate()
        
        audio_bytes = b''.join(frames)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float64)
        audio_np = audio_np / 32768.0
        
        if np.max(np.abs(audio_np)) > 1e-6:
            print("[Parte 2a] Gravacao concluida pelo microfone (PyAudio).")
            return audio_np, fs
            
    except Exception as e:
        print(f"[Parte 2a] Falha ao gravar pelo microfone com PyAudio ({e}); usando sinal sintetico.")

    # --- fallback: sinal sintetico "tipo fala" (soma de formantes com envoltoria) ---
    print("[Parte 2a] Nenhum microfone disponivel: gerando sinal sintetico de teste.")
    t = np.arange(0, duracao, 1 / fs)
    formantes = [300, 800, 1800, 2600]
    audio = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate(formantes))
    envoltoria = np.sin(np.pi * t / duracao) ** 2          
    ruido = 0.01 * np.random.randn(len(t))
    audio = audio * envoltoria + ruido
    audio = audio / np.max(np.abs(audio))                  
    return audio, fs

def amostragem(audio, fs_original, fs_amostragem):
    """
    (b) Amostragem.

    Reamostra o sinal (ja discreto, pois veio de uma placa de som) para
    a taxa 'fs_amostragem' escolhida, simulando o efeito de amostrar o
    sinal continuo original a essa taxa. Usamos scipy.signal.resample,
    que reamostra no dominio da frequencia (equivalente a amostragem
    ideal seguida de filtragem anti-aliasing).
    """
    n_amostras = int(len(audio) * fs_amostragem / fs_original)
    audio_amostrado = sig.resample(audio, n_amostras)
    return audio_amostrado, fs_amostragem


def quantizacao(audio, n_bits):
    """
    (c) Quantizacao uniforme com 2**n_bits niveis, cobrindo a faixa
    dinamica real do sinal (min a max).
    """
    n_niveis = 2 ** n_bits
    x_min, x_max = audio.min(), audio.max()
    passo = (x_max - x_min) / (n_niveis - 1)

    indices = np.round((audio - x_min) / passo).astype(int)
    indices = np.clip(indices, 0, n_niveis - 1)
    audio_quantizado = x_min + indices * passo             # sinal quantizado (para plot/audio)
    return audio_quantizado, indices, passo


def codificacao(indices, n_bits):
    """
    (d) Codificacao PCM: cada indice de quantizacao vira uma palavra
    binaria de n_bits bits.
    """
    codigos = [format(i, f"0{n_bits}b") for i in indices]
    return codigos


def decodificar(codigos, x_min, passo):
    """Reconstroi o sinal a partir das palavras binarias (para o
    calculo do erro/SNR e para gerar o .wav de saida)."""
    indices = np.array([int(c, 2) for c in codigos])
    return x_min + indices * passo


def parte2_pcm(fs_gravacao=16000, fs_amostragem=8000, n_bits=8, zoom_ms=20):
    audio, fs0 = gravar_ou_gerar_audio(duracao=2.0, fs=fs_gravacao)
    t0 = np.arange(len(audio)) / fs0

    # salva o audio de entrada
    wavfile.write(f"{OUT_DIR}/2a_entrada.wav", fs0, audio.astype(np.float32))

    # (a) plot do sinal original
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t0, audio, lw=0.8)
    ax.set_title("(a) Sinal de audio original")
    ax.set_xlabel("tempo (s)"); ax.set_ylabel("amplitude"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/2a_original.png", dpi=140); plt.close(fig)

    # (b) amostragem
    audio_amostrado, fs1 = amostragem(audio, fs0, fs_amostragem)
    t1 = np.arange(len(audio_amostrado)) / fs1
    wavfile.write(f"{OUT_DIR}/2b_amostrado.wav", fs1, audio_amostrado.astype(np.float32))

    n_zoom = int(zoom_ms / 1000 * fs1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t0[: int(zoom_ms / 1000 * fs0)], audio[: int(zoom_ms / 1000 * fs0)],
            label="original", alpha=0.5)
    ax.stem(t1[:n_zoom], audio_amostrado[:n_zoom], linefmt="C1-", markerfmt="C1o",
            basefmt=" ", label=f"amostrado @ {fs_amostragem} Hz")
    ax.set_title(f"(b) Amostragem (zoom de {zoom_ms} ms)")
    ax.set_xlabel("tempo (s)"); ax.set_ylabel("amplitude"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/2b_amostrado.png", dpi=140); plt.close(fig)

    # (c) quantizacao
    audio_quantizado, indices, passo = quantizacao(audio_amostrado, n_bits)
    wavfile.write(f"{OUT_DIR}/2c_quantizado.wav", fs1, audio_quantizado.astype(np.float32))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(t1[:n_zoom], audio_quantizado[:n_zoom], where="mid", color="tab:red",
            label=f"quantizado ({2**n_bits} niveis)")
    ax.plot(t1[:n_zoom], audio_amostrado[:n_zoom], color="tab:orange", alpha=0.4, label="amostrado")
    ax.set_title(f"(c) Quantizacao ({n_bits} bits -> {2**n_bits} niveis, zoom {zoom_ms} ms)")
    ax.set_xlabel("tempo (s)"); ax.set_ylabel("amplitude"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/2c_quantizado.png", dpi=140); plt.close(fig)

    # (d) codificacao
    codigos = codificacao(indices, n_bits)
    print(f"[Parte 2d] Exemplo de palavras-codigo (primeiras 10): {codigos[:10]}")
    print(f"[Parte 2d] Taxa de bits do PCM: {fs_amostragem * n_bits} bits/s "
          f"({fs_amostragem} amostras/s x {n_bits} bits/amostra)")

    audio_decodificado = decodificar(codigos, audio_amostrado.min(), passo)
    wavfile.write(f"{OUT_DIR}/2d_codificado_reconstruido.wav", fs1,
                  audio_decodificado.astype(np.float32))

    # (e) comparacao / erro de quantizacao
    erro = audio_amostrado - audio_decodificado
    ruido_quant_rms = np.sqrt(np.mean(erro ** 2))
    sinal_rms = np.sqrt(np.mean(audio_amostrado ** 2))
    snr_db = 20 * np.log10(sinal_rms / ruido_quant_rms) if ruido_quant_rms > 0 else np.inf
    print(f"[Parte 2e] SNR de quantizacao ({n_bits} bits, fs={fs_amostragem} Hz): "
          f"{snr_db:.2f} dB")

    return {
        "fs_amostragem": fs_amostragem, "n_bits": n_bits,
        "snr_db": snr_db, "ruido_rms": ruido_quant_rms,
    }


def comparar_parametros():
    """
    (e) Varia a taxa de amostragem e o numero de bits por amostra e
    compara o SNR de quantizacao resultante -- mostra que:
      - mais bits por amostra => menor ruido de quantizacao => maior SNR;
      - a taxa de amostragem, por si so, nao muda o SNR de quantizacao,
        mas taxas baixas demais (abaixo de 2x a maior frequencia do
        sinal) introduzem aliasing e degradam a inteligibilidade.
    """
    resultados = []
    for fs_amostragem in [4000, 8000, 16000]:
        for n_bits in [4, 8, 12]:
            r = parte2_pcm(fs_amostragem=fs_amostragem, n_bits=n_bits)
            resultados.append(r)

    print("\n[Comparacao] fs_amostragem | n_bits | SNR (dB)")
    for r in resultados:
        print(f"  {r['fs_amostragem']:>10} | {r['n_bits']:>6} | {r['snr_db']:.2f}")


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("PARTE 2 - PCM (execucao principal, 8 bits / 8 kHz)")
    print("=" * 60)
    parte2_pcm(fs_gravacao=16000, fs_amostragem=8000, n_bits=8)

    print("\n" + "=" * 60)
    print("PARTE 2e - Comparando taxas de amostragem e bits")
    print("=" * 60)
    comparar_parametros()

    print(f"\nTodos os graficos e arquivos .wav foram salvos em ./{OUT_DIR}/")
