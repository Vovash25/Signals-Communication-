import numpy as np
import matplotlib.pyplot as plt


fs = 10000
fc = 1000
T_bit = 0.002
bits = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1]) 

def generate_carrier(A, phi, t_segment):
    # x(t) = A * sin(2*pi*f*t + phi)
    return A * np.sin(2 * np.pi * fc * t_segment + phi)


def mod_bpsk(dane):
    t_seg = np.linspace(0, T_bit, int(fs * T_bit))
    sygnal = []
    for b in dane:
        phi = np.pi if b == 1 else 0
        sygnal.extend(generate_carrier(1.0, phi, t_seg))
    return np.array(sygnal)

def mod_qpsk(dane):
    t_seg = np.linspace(0, T_bit * 2, int(fs * T_bit * 2))
    sygnal = []
    mapa = {(0,1): np.pi, (0,0): 0, (1,1): np.pi/2, (1,0): -np.pi/2}
    for i in range(0, len(dane), 2):
        para = tuple(dane[i:i+2])
        sygnal.extend(generate_carrier(1.0, mapa[para], t_seg))
    return np.array(sygnal)

def mod_qam8(dane):
    t_seg = np.linspace(0, T_bit * 3, int(fs * T_bit * 3))
    sygnal = []
    mapa = {
        (0,0,0): (1.0, 0), (0,0,1): (1.0, np.pi/2), (0,1,0): (1.0, np.pi), (0,1,1): (1.0, 3*np.pi/2),
        (1,0,0): (0.5, 0), (1,0,1): (0.5, np.pi/2), (1,1,0): (0.5, np.pi), (1,1,1): (0.5, 3*np.pi/2)
    }
    for i in range(0, len(dane), 3):
        trojka = tuple(dane[i:i+3])
        A, phi = mapa[trojka]
        sygnal.extend(generate_carrier(A, phi, t_seg))
    return np.array(sygnal)


def mod_ask(dane):
    t_seg = np.linspace(0, T_bit, int(fs * T_bit))
    sygnal = []
    for b in dane:
        sygnal.extend(generate_carrier(1.0 if b == 1 else 0.2, 0, t_seg))
    return np.array(sygnal)

def mod_fsk(dane):
    t_seg = np.linspace(0, T_bit, int(fs * T_bit))
    sygnal = []
    for b in dane:
        f = fc * 2 if b == 1 else fc
        sygnal.extend(np.sin(2 * np.pi * f * t_seg))
    return np.array(sygnal)


def get_psd_db(sig):
    N = len(sig)
    fft_v = np.fft.fft(sig)
    psd = (np.abs(fft_v)**2) / (N * fs)
    freqs = np.fft.fftfreq(N, 1/fs)
    return freqs[:N//2], 10 * np.log10(psd[:N//2] + 1e-15)

def plot_repository_results():
    plt.figure(figsize=(12, 10))
    
    s_qam8 = mod_qam8(bits)
    t_axis = np.arange(len(s_qam8)) / fs
    plt.subplot(2, 1, 1)
    plt.plot(t_axis, s_qam8, 'k-', linewidth=1)
    for x in range(0, len(s_qam8), int(fs * T_bit * 3)):
        plt.axvline(x/fs, color='k', linestyle=':', alpha=0.5)
    plt.title("Przebieg czasowy QAM-8")
    plt.grid(True, linestyle='--')

    # Porównanie  dB
    plt.subplot(2, 1, 2)
    for s, label in [(s_qam8, "QAM-8"), (mod_bpsk(bits), "PSK (BPSK)"), 
                    (mod_ask(bits), "ASK"), (mod_fsk(bits), "FSK")]:
        f, p = get_psd_db(s)
        plt.plot(f, p, label=label)
    
    plt.title("Porownanie widm gestosci mocy (PSD) [dB]")
    plt.xlim(0, fc * 4)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_repository_results()