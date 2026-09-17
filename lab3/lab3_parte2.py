
"""
Lab 3 - Parte 2: PCM (amostragem, quantizacao, codificacao)
=============================================================
Correcoes aplicadas em relacao a versao anterior:
  1) import de 'pyaudio' agora e protegido (try/except), assim como o
     de 'sounddevice' -- se nenhuma das duas bibliotecas de audio
     estiver instalada/funcionando, o script nao quebra: ele cai para
     um sinal sintetico automaticamente.
  2) (a) agora REPRODUZ o audio gravado (requisito do PDF: "utilize
     uma funcao da ferramenta para escuta-lo"), usando sounddevice
     quando disponivel.
  3) (d) agora tambem gera um GRAFICO do sinal codificado/decodificado
     (2d_codificado.png), alem do .wav -- antes so existia o .wav.
  4) Imports mortos removidos ('wave', 'keyboard', 'time', que nao
     eram usados em nenhum lugar do codigo).
  5) A gravacao agora acontece UMA UNICA VEZ (no bloco principal), e o
     item (e) reprocessa esse MESMO audio nas 9 combinacoes de taxa de
     amostragem x bits. Antes, 'comparar_parametros()' chamava
     'gravar_ou_gerar_audio()' de novo a cada uma das 9 combinacoes,
     cada uma esperando ENTER + Ctrl+C -- se essa interacao nao
     ocorresse (ou o Ctrl+C nao fosse entregue ao processo, o que e
     comum em notebooks/IDEs), o script ficava preso indefinidamente
     dentro do laco de gravacao logo na primeira chamada.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal as sig

# --- audio: PyAudio (gravacao manual com Ctrl+C) e sounddevice (playback
# e gravacao de duracao fixa como alternativa) sao ambos opcionais.
# Se nenhum estiver disponivel no ambiente, o script usa um sinal
# sintetico no lugar da gravacao real, so para nao travar a execucao. ---
try:
    import pyaudio
    PYAUDIO_OK = True
except Exception:
    PYAUDIO_OK = False

try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except Exception:
    SOUNDDEVICE_OK = False

OUT_DIR = "saida_lab3"
os.makedirs(OUT_DIR, exist_ok=True)


def reproduzir_audio(audio, fs, rotulo=""):
    """Toca o audio pela saida de som (requisito: 'utilize uma funcao
    da ferramenta para escuta-lo'). Se nao houver dispositivo de saida
    de audio disponivel, apenas avisa e segue em frente."""
    if not SOUNDDEVICE_OK:
        print(f"[audio] sounddevice indisponivel: nao foi possivel reproduzir {rotulo}.")
        return
    try:
        print(f"[audio] Reproduzindo {rotulo}...")
        sd.play(audio, fs)
        sd.wait()
    except Exception as e:
        print(f"[audio] Nao foi possivel reproduzir {rotulo} ({e}).")


def gravar_ou_gerar_audio(duracao=2.0, fs=16000):
    """
    (a) Gravacao do audio de entrada.

    Tenta gravar pelo microfone com PyAudio (gravacao manual: aperte
    ENTER para comecar e Ctrl+C para parar -- util para gravar uma
    frase de tamanho livre). Se o PyAudio nao estiver disponivel, tenta
    o sounddevice com duracao fixa. Se nenhum dispositivo de audio
    existir no ambiente, gera um sinal sintetico apenas para que o
    restante da cadeia PCM possa ser demonstrado.
    """
    if PYAUDIO_OK:
        try:
            FORMAT = pyaudio.paInt16
            CHUNK = 1024
            audio_obj = pyaudio.PyAudio()

            input(f"[Parte 2a] Pressione ENTER para comecar a gravar a {fs} Hz...")
            stream = audio_obj.open(format=FORMAT, channels=1, rate=fs,
                                     input=True, frames_per_buffer=CHUNK)

            frames = []
            print("[Parte 2a] Gravando... Pressione Ctrl+C para parar.")
            while True:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except KeyboardInterrupt:
                    print("\n[Parte 2a] Parando gravacao...")
                    break

            stream.stop_stream()
            stream.close()
            audio_obj.terminate()

            audio_bytes = b"".join(frames)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float64)
            audio_np = audio_np / 32768.0

            if np.max(np.abs(audio_np)) > 1e-6:
                print("[Parte 2a] Gravacao concluida pelo microfone (PyAudio).")
                return audio_np, fs
        except Exception as e:
            print(f"[Parte 2a] Falha ao gravar com PyAudio ({e}); tentando outra opcao.")

    if SOUNDDEVICE_OK:
        try:
            print(f"[Parte 2a] Gravando {duracao}s a {fs} Hz (sounddevice)... fale a frase agora.")
            audio = sd.rec(int(duracao * fs), samplerate=fs, channels=1, dtype="float64")
            sd.wait()
            audio = audio.flatten()
            if np.max(np.abs(audio)) > 1e-6:
                print("[Parte 2a] Gravacao concluida pelo microfone (sounddevice).")
                return audio, fs
        except Exception as e:
            print(f"[Parte 2a] Falha ao gravar com sounddevice ({e}); usando sinal sintetico.")

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
    audio_quantizado = x_min + indices * passo
    return audio_quantizado, indices, passo


def codificacao(indices, n_bits):
    """
    (d) Codificacao PCM: cada indice de quantizacao vira uma palavra
    binaria de n_bits bits.
    """
    codigos = [format(i, f"0{n_bits}b") for i in indices]
    return codigos


def desenhar_niveis_quantizacao(ax, x_min, passo, n_niveis, margem_niveis=5, max_linhas=40):
    """
    Desenha uma linha horizontal tracejada em cada nivel de quantizacao
    disponivel, para que fique visivel em que "degraus" o sinal pode
    ficar.

    So desenha os niveis proximos da faixa vertical que os dados
    plotados ocupam, e ainda assim adiciona uma MARGEM de alguns
    niveis acima/abaixo (parametro 'margem_niveis'): sem essa margem,
    se o trecho de zoom oscilar pouco (ficar "grudado" em 1 ou 2
    niveis), as linhas de referencia cairiam exatamente sobre a propria
    curva plotada e ficariam escondidas atras dela. Com a margem, o
    grafico sempre mostra alguns niveis vazios acima e abaixo do sinal,
    deixando claro o "degrau" entre eles.

    Com poucos bits (poucos niveis, bem espacados) todas as linhas
    aparecem nitidamente; com muitos bits (niveis muito proximos),
    'max_linhas' evita que o grafico vire uma mancha solida.
    """
    y0, y1 = ax.get_ylim()
    y0 -= margem_niveis * passo
    y1 += margem_niveis * passo

    primeiro = max(0, int(np.floor((y0 - x_min) / passo)))
    ultimo = min(n_niveis - 1, int(np.ceil((y1 - x_min) / passo)))
    indices = list(range(primeiro, ultimo + 1))

    if len(indices) > max_linhas:
        passo_amostragem = max(1, len(indices) // max_linhas)
        indices = indices[::passo_amostragem]

    for i in indices:
        ax.axhline(x_min + i * passo, color="0.5", lw=0.7, alpha=0.6,
                   linestyle="--", zorder=0)

    ax.set_ylim(y0, y1)  # aplica a janela com a margem, para as linhas ficarem visiveis


def decodificar(codigos, x_min, passo):
    """Reconstroi o sinal a partir das palavras binarias (para plot,
    calculo do erro/SNR e para gerar o .wav de saida)."""
    indices = np.array([int(c, 2) for c in codigos])
    return x_min + indices * passo


def parte2_pcm(audio, fs0, fs_amostragem=8000, n_bits=8, zoom_ms=20, escutar=False):
    """
    Processa UM audio ja gravado (passado em 'audio'/'fs0') pela cadeia
    PCM completa: amostragem -> quantizacao -> codificacao. Nao grava
    nada aqui -- a gravacao acontece uma unica vez, fora desta funcao,
    para que o item (e) possa reprocessar o MESMO audio com varias
    combinacoes de taxa/bits sem pedir uma nova gravacao a cada vez.
    """
    t0 = np.arange(len(audio)) / fs0

    wavfile.write(f"{OUT_DIR}/2a_entrada.wav", fs0, audio.astype(np.float32))
    if escutar:
        reproduzir_audio(audio, fs0, "sinal original (a)")

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
    if escutar:
        reproduzir_audio(audio_amostrado, fs1, f"sinal amostrado @ {fs_amostragem} Hz (b)")

    n_zoom = max(2, int(zoom_ms / 1000 * fs1))
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
    if escutar:
        reproduzir_audio(audio_quantizado, fs1, f"sinal quantizado {n_bits} bits (c)")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(t1[:n_zoom], audio_quantizado[:n_zoom], where="mid", color="tab:red",
            label=f"quantizado ({2**n_bits} niveis)", zorder=2)
    ax.plot(t1[:n_zoom], audio_amostrado[:n_zoom], color="tab:orange", alpha=0.4,
            label="amostrado", zorder=1)
    desenhar_niveis_quantizacao(ax, audio_amostrado.min(), passo, 2 ** n_bits)
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
    if escutar:
        reproduzir_audio(audio_decodificado, fs1, f"sinal codificado/decodificado {n_bits} bits (d)")

    # plot do sinal codificado (decodificado a partir das palavras binarias),
    # exigido explicitamente no item (d) do roteiro
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(t1[:n_zoom], audio_decodificado[:n_zoom], where="mid", color="tab:purple",
            label=f"codificado ({n_bits} bits/amostra)", zorder=2)
    desenhar_niveis_quantizacao(ax, audio_amostrado.min(), passo, 2 ** n_bits)
    ax.set_title(f"(d) Sinal codificado (PCM, {n_bits} bits/amostra, zoom {zoom_ms} ms)")
    ax.set_xlabel("tempo (s)"); ax.set_ylabel("amplitude"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/2d_codificado.png", dpi=140); plt.close(fig)

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


def comparar_parametros(audio, fs0, escutar=True):
    """
    (e) Pega o MESMO audio gravado no inicio e o reproduz reamostrado
    em cada uma das taxas de amostragem testadas -- assim da pra ouvir
    diretamente o efeito de cada taxa (perda de agudos, ou ate
    aliasing nas taxas mais baixas) antes de olhar os numeros de SNR.

    Em seguida, para cada taxa, ainda varia o numero de bits por
    amostra e calcula o SNR de quantizacao, mostrando que:
      - mais bits por amostra => menor ruido de quantizacao => maior SNR;
      - a taxa de amostragem, por si so, nao muda o SNR de quantizacao,
        mas taxas baixas demais (abaixo de 2x a maior frequencia do
        sinal) introduzem aliasing e degradam a inteligibilidade --
        efeito que se ouve, mesmo sem aparecer no SNR.
    """
    resultados = []
    for fs_amostragem in [4000, 8000, 16000]:
        # reamostra uma unica vez nesta taxa e reproduz o audio
        # resultante, para poder OUVIR o efeito da taxa de amostragem
        # isoladamente (sem a quantizacao de baixa resolucao atrapalhando)
        audio_amostrado, fs1 = amostragem(audio, fs0, fs_amostragem)
        wavfile.write(f"{OUT_DIR}/2e_amostrado_{fs_amostragem}Hz.wav",
                      fs1, audio_amostrado.astype(np.float32))
        if escutar:
            reproduzir_audio(audio_amostrado, fs1,
                             f"audio reamostrado a {fs_amostragem} Hz (item e)")

        for n_bits in [4, 8, 12]:
            r = parte2_pcm(audio, fs0, fs_amostragem=fs_amostragem, n_bits=n_bits, escutar=False)
            resultados.append(r)

    print("\n[Comparacao] fs_amostragem | n_bits | SNR (dB)")
    for r in resultados:
        print(f"  {r['fs_amostragem']:>10} | {r['n_bits']:>6} | {r['snr_db']:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Gravacao (uma unica vez, reaproveitada em todas as etapas)")
    print("=" * 60)
    audio, fs0 = gravar_ou_gerar_audio(duracao=2.0, fs=16000)

    print("\n" + "=" * 60)
    print("PARTE 2 - PCM (execucao principal, 8 bits / 8 kHz)")
    print("=" * 60)
    # escutar=True toca o audio de cada etapa (a, b, c, d); coloque
    # False se estiver rodando sem caixa de som/fones.
    parte2_pcm(audio, fs0, fs_amostragem=8000, n_bits=8, escutar=True)

    print("\n" + "=" * 60)
    print("PARTE 2e - Comparando taxas de amostragem e bits")
    print("=" * 60)
    # escutar=True reproduz o audio reamostrado a 4000/8000/16000 Hz
    # antes de calcular o SNR de cada taxa; troque para False se
    # estiver rodando sem caixa de som/fones.
    comparar_parametros(audio, fs0, escutar=True)

    print(f"\nTodos os graficos e arquivos .wav foram salvos em ./{OUT_DIR}/")
