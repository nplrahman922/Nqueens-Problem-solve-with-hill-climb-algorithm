import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
from matplotlib import colors

np.random.seed(1234)


def random_board(n):
    """Creates a random board of size n x n. Note that only a single queen is placed in each column!"""

    return(np.random.randint(0,n, size = n))

def comb2(n): return n*(n-1)//2 # this is n choose 2 equivalent to math.comb(n, 2); // is int division

def conflicts(board):
    """Calculate the number of conflicts, i.e., the objective function."""

    n = len(board)

    horizontal_cnt = [0] * n
    diagonal1_cnt = [0] * 2 * n
    diagonal2_cnt = [0] * 2 * n

    for i in range(n):
        horizontal_cnt[board[i]] += 1
        diagonal1_cnt[i + board[i]] += 1
        diagonal2_cnt[i - board[i] + n] += 1

    return sum(map(comb2, horizontal_cnt + diagonal1_cnt + diagonal2_cnt))

# decrease the font size to fit larger boards
def show_board(board, cols = ['white', 'gray'], fontsize = 48):
    """display the board"""

    n = len(board)

    # create chess board display
    display = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if (((i+j) % 2) != 0):
                display[i,j] = 1

    cmap = colors.ListedColormap(cols)
    fig, ax = plt.subplots()
    ax.imshow(display, cmap = cmap,
              norm = colors.BoundaryNorm(range(len(cols)+1), cmap.N))
    ax.set_xticks([])
    ax.set_yticks([])

    # place queens. Note: Unicode u265B is a black queen
    for j in range(n):
        plt.text(j, board[j], u"\u265B", fontsize = fontsize,
                 horizontalalignment = 'center',
                 verticalalignment = 'center')

    print(f"Board with {conflicts(board)} conflicts.")
    plt.show()

# Asumsikan Anda memiliki fungsi conflicts(board) yang sudah didefinisikan
# def conflicts(board):
#     ... (kode untuk menghitung konflik) ...

def steepest_ascent_hill_climb(board):
    """
    Melakukan steepest ascent hill climbing untuk meminimalkan jumlah konflik.
    Versi yang sudah diperbaiki.
    """
    n = len(board)
    current_board = board.copy() # Bekerja dengan salinan agar tidak mengubah input asli
    current_conflicts = conflicts(current_board)

    while True:
        next_board = None
        min_next_conflicts = current_conflicts

        # Jelajahi semua tetangga untuk menemukan yang terbaik (paling curam naiknya)
        for col in range(n):
            for row in range(n):
                # Tidak perlu mengevaluasi state saat ini
                if current_board[col] == row:
                    continue

                # BUAT SALINAN SEMENTARA untuk evaluasi
                temp_board = current_board.copy()
                temp_board[col] = row # Ubah posisi Ratu di salinan
                
                new_conflicts = conflicts(temp_board)

                # Cek apakah tetangga ini adalah yang terbaik sejauh ini
                if new_conflicts < min_next_conflicts:
                    min_next_conflicts = new_conflicts
                    next_board = temp_board

        # Jika tidak ada tetangga yang lebih baik, kita mencapai optimum lokal
        if next_board is None:
            break  # Keluar dari loop

        # Pindah ke state tetangga terbaik
        current_board = next_board
        current_conflicts = min_next_conflicts

    return current_board

def stochastic_hill_climb(board):
    """
    Melakukan stochastic hill climbing untuk meminimalkan jumlah konflik.
    Versi yang sudah diperbaiki.
    """
    n = len(board)
    current_board = board.copy() # Bekerja dengan salinan agar tidak mengubah input asli
    current_conflicts = conflicts(current_board)

    while True:
        better_neighbors = []

        # Jelajahi semua tetangga untuk menemukan yang lebih baik
        for col in range(n):
            for row in range(n):
                # Tidak perlu mengevaluasi state saat ini
                if current_board[col] == row:
                    continue

                # BUAT SALINAN SEMENTARA untuk evaluasi
                temp_board = current_board.copy()
                temp_board[col] = row # Ubah posisi Ratu di salinan
                
                new_conflicts = conflicts(temp_board)

                # Cek apakah tetangga ini lebih baik
                if new_conflicts < current_conflicts:
                    better_neighbors.append(temp_board)

        # Jika tidak ada tetangga yang lebih baik, kita mencapai optimum lokal
        if not better_neighbors:
            break  # Keluar dari loop

        # Pilih secara acak salah satu tetangga yang lebih baik
        current_board = better_neighbors[np.random.randint(len(better_neighbors))]
        current_conflicts = conflicts(current_board)

    return current_board    

def standard_stochastic_hill_climb(board, max_iterations_without_improvement=100):
    """
    Melakukan stochastic hill climbing (versi standar/murni).
    """
    n = len(board)
    current_board = board.copy()
    current_conflicts = conflicts(current_board)
    
    iterations_stuck = 0

    # Loop utama bisa berdasarkan jumlah iterasi atau kondisi berhenti lainnya
    while iterations_stuck < max_iterations_without_improvement:
        if current_conflicts == 0:
            break # Solusi ditemukan

        # 1. Pilih SATU tetangga secara acak
        random_col = random.randint(0, n - 1)
        # Pastikan baris baru berbeda dari yang lama
        current_row = current_board[random_col]
        possible_rows = list(range(n))
        possible_rows.remove(current_row)
        random_row = random.choice(possible_rows)

        # Buat state tetangga
        neighbor_board = current_board.copy()
        neighbor_board[random_col] = random_row
        
        # 2. Evaluasi tetangga tersebut
        neighbor_conflicts = conflicts(neighbor_board)

        # 3. Pindah jika lebih baik
        if neighbor_conflicts < current_conflicts:
            current_board = neighbor_board
            current_conflicts = neighbor_conflicts
            iterations_stuck = 0 # Reset counter karena ada perbaikan
            # print(f"Pindah, konflik baru: {current_conflicts}") # Untuk debugging
        else:
            iterations_stuck += 1 # Tambah counter jika tidak ada perbaikan

    return current_board

def random_restart_hill_climb(board, max_restarts=100):
    """
    Melakukan random restart hill climbing untuk meminimalkan jumlah konflik.
    """
    n = len(board)
    best_board = None
    best_conflicts = float('inf')

    for _ in range(max_restarts):
        # Mulai dengan papan acak
        initial_board = random_board(n)
        # Gunakan steepest ascent hill climbing dari papan acak ini
        final_board = steepest_ascent_hill_climb(initial_board)
        final_conflicts = conflicts(final_board)

        # Periksa apakah ini adalah solusi terbaik sejauh ini
        if final_conflicts < best_conflicts:
            best_conflicts = final_conflicts
            best_board = final_board

        # Jika kita menemukan solusi tanpa konflik, kita bisa berhenti lebih awal
        if best_conflicts == 0:
            break

    return best_board    

def simulated_annealing(board, initial_temp=100, cooling_rate=0.99, max_iterations=1000):
    """
    Melakukan simulated annealing untuk meminimalkan jumlah konflik.
    """
    n = len(board)
    current_board = board.copy()
    current_conflicts = conflicts(current_board)
    temperature = initial_temp

    best_board = current_board.copy()
    best_conflicts = current_conflicts

    for iteration in range(max_iterations):
        if current_conflicts == 0:
            break  # Solusi ditemukan

        # Pilih SATU tetangga secara acak
        random_col = random.randint(0, n - 1)
        current_row = current_board[random_col]
        possible_rows = list(range(n))
        possible_rows.remove(current_row)
        random_row = random.choice(possible_rows)

        neighbor_board = current_board.copy()
        neighbor_board[random_col] = random_row
        neighbor_conflicts = conflicts(neighbor_board)

        delta_conflicts = neighbor_conflicts - current_conflicts

        # Jika tetangga lebih baik, pindah ke sana
        if delta_conflicts < 0:
            current_board = neighbor_board
            current_conflicts = neighbor_conflicts
        else:
            # Jika tetangga lebih buruk, pindah dengan probabilitas tertentu
            acceptance_probability = np.exp(-delta_conflicts / temperature)
            if random.random() < acceptance_probability:
                current_board = neighbor_board
                current_conflicts = neighbor_conflicts

        if current_conflicts < best_conflicts:
            best_conflicts = current_conflicts
            best_board = current_board.copy()
        
        # Pendinginan suhu
        temperature *= cooling_rate

    return best_board    
# example usage

# board = random_board(4)
# show_board(board)
# print(f"Queens (left to right) are at rows: {board}")
# print(f"Number of conflicts: {conflicts(board)}")

# board = simulated_annealing(board)
# show_board(board)
# print(f"Queens (left to right) are at rows: {board}")
# print(f"Number of conflicts: {conflicts(board)}")
algorithms = {
    "Steepest Ascent Hill Climb": steepest_ascent_hill_climb,
    "Stochastic Hill Climb": stochastic_hill_climb,
    "Standard Stochastic Hill Climb": standard_stochastic_hill_climb,
    "Random-Restart Hill Climb": random_restart_hill_climb,
    "Simulated Annealing": simulated_annealing
}

# Definisikan parameter eksperimen
board_sizes = [4, 8]  # Ukuran papan yang akan diuji
num_runs = 100       # Jumlah pengulangan untuk mendapatkan rata-rata yang akurat

# Siapkan list kosong untuk menyimpan semua hasil
results = []

print("Memulai proses perbandingan algoritma...")

# Loop utama untuk menjalankan eksperimen
for size in board_sizes:
    print(f"\nMenguji untuk papan ukuran {size}x{size}...")
    for name, algorithm_func in algorithms.items():
        # Siapkan variabel untuk mengakumulasi hasil setiap kali loop dimulai
        total_time = 0
        total_conflicts = 0
        
        # Jalankan algoritma yang sama sebanyak num_runs kali
        for _ in range(num_runs):
            # 1. Buat papan acak baru untuk setiap eksekusi
            initial_board = random_board(size)
            
            # 2. Ukur waktu mulai
            start_time = time.time()
            
            # 3. Jalankan fungsi algoritma
            final_board = algorithm_func(initial_board)
            
            # 4. Ukur waktu selesai
            end_time = time.time()
            
            # 5. Akumulasi waktu dan jumlah konflik
            total_time += end_time - start_time
            total_conflicts += conflicts(final_board)

        # Hitung rata-rata setelah semua pengulangan selesai
        avg_time = total_time / num_runs
        avg_conflicts = total_conflicts / num_runs
        
        # Simpan hasil rata-rata ke dalam list
        results.append({
            "Algorithm": name,
            "Board Size": size,
            "Average Time (s)": avg_time,
            "Average Conflicts": avg_conflicts
        })
        print(f"  - Selesai menguji: {name}")

print("\nProses perbandingan selesai.")

# ==============================================================================
# BAGIAN 2: TAMPILKAN HASIL MENGGUNAKAN LIBRARY PANDAS
# ==============================================================================

# Buat DataFrame (tabel) dari list hasil yang sudah dikumpulkan
df_results = pd.DataFrame(results)

# Atur format tampilan angka agar lebih mudah dibaca (opsional)
pd.options.display.float_format = '{:.6f}'.format

print("\n--- Hasil Perbandingan Kinerja Algoritma ---")
# Gunakan .to_string() untuk memastikan semua baris dan kolom tercetak
print(df_results.to_string())
