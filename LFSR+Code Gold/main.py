import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
"""
def lfsr(taps, length):
    state = [1] * length
    seq = []
    for _ in range(2**length - 1):
        feedback = 0
        for t in taps: feedback ^= state[t-1]
        seq.append(state[-1])
        state = [feedback] + state[:-1]
    return seq

def generate_gold_code(shift=0):
    s1 = lfsr([5,3], 5)
    s2 = lfsr([5,4,3,2], 5)
    s2_shifted = s2[shift:] + s2[:shift]
    return [a ^ b for a, b in zip(s1, s2_shifted)]
"""
def lfsr(taps, length):
    state = [1] * length
    seq = []
    for _ in range(2**length - 1):
        feedback = 0
        for t in taps: feedback ^= state[t-1]
        seq.append(state[-1])
        state = [feedback] + state[:-1]
    return seq

def get_gold_sequence():
    s1 = lfsr([5,3], 5)
    s2 = lfsr([5,4,3,2], 5)
    return [a ^ b for a, b in zip(s1, s2)]

#Parametry
fs = 12000
T_bit = 0.04
Nc = 4
T_chip = T_bit / Nc
samples_per_chip = int(fs * T_chip)

bits = np.array([1, 0, 1, 1, 0, 1])
f_table = np.linspace(800, 4000, 16) 

def bits_to_idx(seq, start_idx):
    sub = [seq[(start_idx + i) % len(seq)] for i in range(4)]
    res = 0
    for b in sub: res = (res << 1) | b
    return res % 16

#Modulacja
gold_seq = get_gold_sequence()
sig_fsk = []
sig_fhss = []

f0, f1 = 1200, 1800 

for i, bit in enumerate(bits):
    # FSK
    f_current = f1 if bit == 1 else f0
    t_bit = np.linspace(i*T_bit, (i+1)*T_bit, int(fs*T_bit), endpoint=False)
    sig_fsk.extend(np.sin(2 * np.pi * f_current * t_bit))
    for c in range(Nc):
        hop_idx = bits_to_idx(gold_seq, (i * Nc + c) * 4)
        f_hop = f_table[hop_idx]
        
        f_final = f_hop + (200 if bit == 1 else 0)
        
        t_chip = np.linspace((i*Nc + c)*T_chip, (i*Nc + c + 1)*T_chip, samples_per_chip, endpoint=False)
        sig_fhss.extend(np.sin(2 * np.pi * f_final * t_chip))

sig_fsk = np.array(sig_fsk)
sig_fhss = np.array(sig_fhss)


# bin.png
plt.figure(figsize=(10, 2))
plt.step(np.arange(len(bits)), bits, where='post', color='blue')
plt.ylim(-0.2, 1.2)
plt.title("Binary Input Data (bin)")
plt.grid(True)
plt.savefig("bin.png")

# fsk.png - FSK
plt.figure(figsize=(10, 3))
plt.plot(sig_fsk[:int(fs*T_bit*2)], color='green')
plt.title("FSK Time Domain (fsk)")
plt.savefig("fsk.png")

# fsk_ss.png - FSK(FHSS)
plt.figure(figsize=(10, 3))
plt.plot(sig_fhss[:int(fs*T_bit*2)], color='red')
plt.title("FHSS Time Domain (fsk_ss)")
plt.savefig("fsk_ss.png")

# fft.png i fft_ss.png 
def save_spectrum(signal, filename, title):
    plt.figure(figsize=(10, 4))
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
    magnitude = np.abs(np.fft.fft(signal))[:N//2]
    plt.plot(freqs, magnitude)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.grid(True)
    plt.savefig(filename)

save_spectrum(sig_fsk, "fft.png", "Spectrum FSK (No Spreading)")
save_spectrum(sig_fhss, "fft_ss.png", "Spectrum FHSS (With Spreading)")

# f_tab.txt - tabela
with open("f_tab.txt", "w") as f:
    f.write("Index | Frequency (Hz)\n" + "-"*20 + "\n")
    for i, freq in enumerate(f_table):
        f.write(f"{i:5} | {freq:10.1f}\n")

print("Wszystkie pliki zostały pomyślnie wygenerowane.")