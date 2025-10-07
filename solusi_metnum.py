# TUGAS METODE NUMERIK: PENYELESAIAN SISTEM PERSAMAAN NON-LINEAR
# File ini mendemonstrasikan 3 metode untuk menyelesaikan sistem:
# f1(x, y) = x^2 + xy - 10 = 0
# f2(x, y) = y + 3xy^2 - 57 = 0
# Metode yang digunakan:
# 1. Iterasi Titik Tetap (sesuai NIMx = 3, yaitu g1B dan g2B)
# 2. Newton-Raphson
# 3. Secant

import numpy as np

# --- Bagian 0: Definisi Fungsi Utama dan Parameter Awal ---

def f1(x, y):
    """Definisi fungsi pertama f1(x, y)."""
    return x**2 + x * y - 10

def f2(x, y):
    """Definisi fungsi kedua f2(x, y)."""
    return y + 3 * x * y**2 - 57

# Parameter awal yang diberikan dalam soal
X0 = 1.5
Y0 = 3.5
TOLERANSI = 0.000001
MAX_ITER = 50 # Batas maksimum iterasi

# --- Bagian 1: Metode Iterasi Titik Tetap (untuk NIMx = 3) ---

def g1B(x, y):
    """Fungsi iterasi g1B, diturunkan dari f2."""
    if y == 0: return float('inf')
    return (57 - y) / (3 * y**2)

def g2B(x, y):
    """Fungsi iterasi g2B, diturunkan dari f1."""
    if x == 0: return float('inf')
    return (10 - x**2) / x

def solve_fixed_point_divergent(x0, y0, max_iter):
    """Mendemonstrasikan metode Iterasi Titik Tetap (Seidel) untuk NIMx=3."""
    print("=" * 60)
    print("1. HASIL METODE ITERASI TITIK TETAP (Seidel - g1B, g2B)")
    print("-" * 60)
    print(f"{'Iterasi':<10}{'x':>15}{'y':>20}")
    print("-" * 60)

    x, y = x0, y0

    for i in range(1, max_iter + 1):
        x_old, y_old = x, y
        try:
            x = g1B(x_old, y_old)
            y = g2B(x, y_old)
            if np.isnan(x) or np.isinf(x) or abs(y) > 1e6:
                print(f"{i:<10}{x:15.4f}{y:20.4f}")
                print("\n Analisis: Iterasi dihentikan karena nilai divergen (tidak konvergen).")
                return
        except (OverflowError, ZeroDivisionError):
            print("\n Analisis: Iterasi divergen karena terjadi overflow atau pembagian dengan nol.")
            return
        print(f"{i:<10}{x:15.4f}{y:20.4f}")
        # Hanya menampilkan beberapa iterasi awal untuk menunjukkan divergensi
        if i >= 4:
            print("\n Analisis: Iterasi dihentikan karena nilai divergen (tidak konvergen).")
            return

# --- Bagian 2: Metode Newton-Raphson ---

def jacobian_analytic(x, y):
    """Menghitung Matriks Jacobian secara analitik."""
    return np.array([
        [2 * x + y, x],
        [3 * y**2, 1 + 6 * x * y]
    ])

def solve_newton_raphson(x0, y0, tol, max_iter):
    """Menyelesaikan sistem menggunakan metode Newton-Raphson."""
    print("\n" + "=" * 60)
    print("2. HASIL METODE NEWTON-RAPHSON")
    print("-" * 60)
    print(f"{'Iterasi':<10}{'x':>15}{'y':>20}{'Error Relatif':>15}")
    print("-" * 60)

    x, y = x0, y0

    for i in range(1, max_iter + 1):
        J = jacobian_analytic(x, y)
        F = np.array([f1(x, y), f2(x, y)])
        delta = np.linalg.solve(J, -F)
        x_new, y_new = x + delta[0], y + delta[1]
        error = np.max(np.abs((np.array([x_new, y_new]) - np.array([x, y])) / np.array([x_new, y_new])))
        x, y = x_new, y_new
        print(f"{i:<10}{x:15.7f}{y:20.7f}{error:15.7f}")
        if error < tol:
            print("-" * 60)
            print(f"Hasil: Konvergensi tercapai setelah {i} iterasi.")
            print(f"Solusi: x = {x:.7f}, y = {y:.7f}")
            return

# --- Bagian 3: Metode Secant ---

def jacobian_numeric(x, y, h=1e-6):
    """Menghitung Matriks Jacobian secara numerik."""
    J = np.zeros((2, 2))
    f_xy = np.array([f1(x, y), f2(x, y)])
    J[:, 0] = (np.array([f1(x + h, y), f2(x + h, y)]) - f_xy) / h
    J[:, 1] = (np.array([f1(x, y + h), f2(x, y + h)]) - f_xy) / h
    return J

def solve_secant(x0, y0, tol, max_iter):
    """Menyelesaikan sistem menggunakan metode Secant."""
    print("\n" + "=" * 60)
    print("3. HASIL METODE SECANT")
    print("-" * 60)
    print(f"{'Iterasi':<10}{'x':>15}{'y':>20}{'Error Relatif':>15}")
    print("-" * 60)

    x, y = x0, y0

    for i in range(1, max_iter + 1):
        J = jacobian_numeric(x, y)
        F = np.array([f1(x, y), f2(x, y)])
        delta = np.linalg.solve(J, -F)
        x_new, y_new = x + delta[0], y + delta[1]
        error = np.max(np.abs((np.array([x_new, y_new]) - np.array([x, y])) / np.array([x_new, y_new])))
        x, y = x_new, y_new
        print(f"{i:<10}{x:15.7f}{y:20.7f}{error:15.7f}")
        if error < tol:
            print("-" * 60)
            print(f"Hasil: Konvergensi tercapai setelah {i} iterasi.")
            print(f"Solusi: x = {x:.7f}, y = {y:.7f}")
            return

# --- Main Program Execution ---
if __name__ == "__main__":
    solve_fixed_point_divergent(X0, Y0, MAX_ITER)
    solve_newton_raphson(X0, Y0, TOLERANSI, MAX_ITER)
    solve_secant(X0, Y0, TOLERANSI, MAX_ITER)
    print("\n" + "=" * 60)
    print("DEMONSTRASI SELESAI")
    print("=" * 60)
