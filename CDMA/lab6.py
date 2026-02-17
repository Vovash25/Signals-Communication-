import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_binary_file(filename):
    data = []
    if not os.path.exists(filename):
        print(f"Plik: {filename} nie istneje!")
        return None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                row = [1 if char == '1' else -1 for char in line]
                data.append(row)
    return data

def format_list(lst):
    return "(" + ", ".join([f"{x:+d}" if x != 0 else "0" for x in lst]) + ")"

def main():
    codes = load_binary_file("kody.txt")
    messages = load_binary_file("dane.txt")

    if not codes or not messages:
        return

    num_clients = len(codes)

    print("--- Przypisane kody (c_n) ---")
    for i in range(num_clients):
        print(f"Klient-{i+1} (kod): {format_list(codes[i])}")

    print("\n--- Wiadomości (d_n) do przesłania ---")
    for i in range(num_clients):
        print(f"Klient-{i+1} (dane): {format_list(messages[i])}")

    encoded_signals = []
    print("\n--- Zakodowane dane (e_n) ---")
    for i in range(num_clients):
        e_n = []
        for bit in messages[i]:
            e_n.extend([bit * c for c in codes[i]])
        encoded_signals.append(e_n)
        print(f"Klient-{i+1} (zakodowane): {format_list(e_n)}")

    signal_length = len(encoded_signals[0])
    S = [0] * signal_length
    for e_n in encoded_signals:
        for j in range(signal_length):
            S[j] += e_n[j]
    
    print(f"\nSygnał S: {tuple(S)}")

    print("\n--- Odkodowane informacje (d_hat) ---")
    for i in range(num_clients):
        decoded_bits = []
        code_len = len(codes[i])
        
        for j in range(0, len(S), code_len):
            chunk = S[j : j + code_len]
            scalar_product = sum(chunk[k] * codes[i][k] for k in range(code_len))
            decoded_bit = 1 if scalar_product > 0 else -1
            decoded_bits.append(decoded_bit)
            
        print(f"Klient-{i+1} (odkodowane): {format_list(decoded_bits)}")

if __name__ == "__main__":
    main()