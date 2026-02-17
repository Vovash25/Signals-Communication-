import numpy as np
import matplotlib.pyplot as plt
import os


def symulacja_modulacji():
    # Parametry sygnału
    fs = 10000
    fc = 1000
    T_bit = 0.002 
    bits = np.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0])

    def gen_segment(A, phi, t_axis):
        # x(t) = A * sin(2*pi*f*t + phi)
        return A * np.sin(2 * np.pi * fc * t_axis + phi)

    #BPSK
    t_bpsk = np.linspace(0, T_bit, int(fs * T_bit))
    sig_bpsk = []
    for b in bits:
        phi = np.pi if b == 1 else 0
        sig_bpsk.extend(gen_segment(1.0, phi, t_bpsk))

    #QPSK
    t_qpsk = np.linspace(0, T_bit * 2, int(fs * T_bit * 2))
    sig_qpsk = []
    mapa_qpsk = {(0,1): np.pi, (0,0): 0, (1,1): np.pi/2, (1,0): -np.pi/2}
    for i in range(0, len(bits), 2):
        para = tuple(bits[i:i+2])
        sig_qpsk.extend(gen_segment(1.0, mapa_qpsk[para], t_qpsk))

    #QAM-8
    t_qam8 = np.linspace(0, T_bit * 3, int(fs * T_bit * 3))
    sig_qam8 = []
    mapa_qam8 = {
        (0,0,0): (1.0, 0), (0,0,1): (1.0, np.pi/2), (0,1,0): (1.0, np.pi), (0,1,1): (1.0, 3*np.pi/2),
        (1,0,0): (0.5, 0), (1,0,1): (0.5, np.pi/2), (1,1,0): (0.5, np.pi), (1,1,1): (0.5, 3*np.pi/2)
    }
    for i in range(0, len(bits), 3):
        trojka = tuple(bits[i:i+3])
        A, phi = mapa_qam8[trojka]
        sig_qam8.extend(gen_segment(A, phi, t_qam8))

    # --- WIZUALIZACJA
    def rysuj_czasowy(sygnal, n_bits, tytul):
        plt.figure(figsize=(12, 4))
        plt.plot(sygnal, 'k-', linewidth=1) 
        samples_per_sym = int(fs * T_bit * n_bits)
        for x in range(0, len(sygnal), samples_per_sym):
            plt.axvline(x, color='k', linestyle=':', alpha=0.5)
        plt.title(tytul)
        plt.grid(True, linestyle='--')
        plt.show()

    rysuj_czasowy(sig_qam8, 3, "Czas QAM-8")

    # --- ANALIZA WIDMOWA
    def plot_psd(syg, lbl):
        N = len(syg)
        fft_v = np.fft.fft(syg)
        psd = (np.abs(fft_v)**2) / (N * fs)
        freqs = np.fft.fftfreq(N, 1/fs)
        plt.plot(freqs[:N//2], 10 * np.log10(psd[:N//2] + 1e-15), label=lbl)

    plt.figure(figsize=(10, 5))
    plot_psd(sig_bpsk, "BPSK")
    plot_psd(sig_qam8, "QAM-8")
    plt.title("Skala PSD")
    plt.xlim(0, 3000)
    plt.legend()
    plt.grid(True)
    plt.show()

def generatory_kodow():
    def lfsr(taps, length):
        state = [1] * length
        seq = []
        for _ in range(2**length - 1):
            feedback = 0
            for t in taps: feedback ^= state[t-1]
            seq.append(state[-1])
            state = [feedback] + state[:-1]
        return seq

    s1 = lfsr([5,3], 5)
    s2 = lfsr([5,4,3,2], 5)
    
    with open("kody_golda.txt", "w") as f:
        f.write("Pary wielomianow: [5,3] i [5,4,3,2]\n")
        for shift in range(len(s2)):
            s2_shift = s2[shift:] + s2[:shift]
            gold = [a ^ b for a, b in zip(s1, s2_shift)]
            # Sprawdzenie zbalansowania
            bal = "ZBALANSOWANY" if abs(sum(gold) - (len(gold)-sum(gold))) == 1 else "NIEZBALANSOWANY"
            f.write(f"{bal}: {''.join(map(str, gold))}\n")

    def hadamard(n):
        if n == 1: return np.array([[1]])
        h_prev = hadamard(n // 2)
        return np.block([[h_prev, h_prev], [h_prev, -h_prev]])

    h_128 = hadamard(128)
    with open("kody_walsha.txt", "w") as f:
        for row in h_128:
            binary = [1 if x == 1 else 0 for x in row]
            f.write(''.join(map(str, binary)) + "\n")
    
    print("Kody zapisane do plików tekstowych.")
    print(f"Twoje pliki zostały zapisane w: {os.getcwd()}")


if __name__ == "__main__": 
    #symulacja_modulacji()
    generatory_kodow()