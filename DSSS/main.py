import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft, fftfreq, fftshift

# --- 1. GENERATORY KODU ---

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
    gold = [a ^ b for a, b in zip(s1, s2)]
    return np.array([1 if x == 0 else -1 for x in gold])

# --- 2. PARAMETRY SYMULACJI ---
fs = 100000
fc = 5000
bit_rate = 500
N_bits = 10

gold_seq = get_gold_sequence()
Nc = len(gold_seq)
Tb = 1 / bit_rate 
Tc = Tb / Nc

samples_per_chip = int(fs * Tc)
samples_per_bit = samples_per_chip * Nc
t = np.arange(N_bits * samples_per_bit) / fs


#Sygnał bipolarny
bits = np.random.choice([1, -1], N_bits)
dt_signal = np.repeat(bits, samples_per_bit)

pn_sequence = np.tile(gold_seq, N_bits)
pn_signal = np.repeat(pn_sequence, samples_per_chip)
tx_b = dt_signal * pn_signal

# BPSK
carrier = np.cos(2 * np.pi * fc * t)
tx = tx_b * carrier

tx_prim = dt_signal * carrier

def calc_psd(signal, fs):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)
    return fftshift(xf), fftshift(np.abs(yf)**2 / N)

xf, psd_tx = calc_psd(tx, fs)
_, psd_tx_prim = calc_psd(tx_prim, fs)

def save_plot(name, data, title, is_binary=True):
    plt.figure(figsize=(10, 4))
    if is_binary:
        plt.step(t[:samples_per_bit*2], data[:samples_per_bit*2], where='post')
    else:
        plt.plot(t[:samples_per_bit*2], data[:samples_per_bit*2])
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"{name}.png")
    plt.close()

save_plot('d', dt_signal, 'Sygnał wejściowy d_t')
save_plot('pn', pn_signal, 'Kod rozpraszający pn_t (Gold)')
save_plot('tx_b', tx_b, 'Rozproszony sygnał binarny tx_b')
save_plot('tx', tx, 'Przebieg czasowy sygnału tx (DSSS BPSK)', is_binary=False)

# Pliki widma
plt.figure(figsize=(10, 5))
plt.semilogy(xf, psd_tx)
plt.title("Widmo sygnału tx (DSSS)")
plt.xlim(-15000, 15000)
plt.savefig('tx_widmo.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.semilogy(xf, psd_tx_prim)
plt.title("Widmo sygnału BPSK bez rozproszenia")
plt.xlim(-15000, 15000)
plt.savefig('tx_prim_widmo.png')
plt.close()

print("Symulacja zakończona. Wygenerowano pliki graficzne.")