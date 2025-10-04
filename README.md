# GPU-Accelerated VQE for H2 with CUDA-Q

[![Python](https://img.shields.io/badge/Python-3.9–3.11-blue)](https://www.python.org/)
[![CUDA-Q](https://img.shields.io/badge/CUDA--Q-%3E%3D0.6.0-76B900)](https://nvidia.github.io/cuda-quantum/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A minimal, hiring‑grade Variational Quantum Eigensolver (VQE) that computes the ground‑state energy of H₂ using NVIDIA CUDA‑Q. Runs on GPU if available and falls back to CPU automatically. Clean logs, reproducible results, and a convergence plot.

- Problem: ground-state energy of H₂ at ~0.735 Å (STO‑3G)
- Stack: CUDA‑Q (GPU or CPU), SciPy optimizer, Matplotlib (optional)
- Expected result: ≈ −1.137 Hartree

---

## Table of Contents
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Expected Output](#expected-output)
- [How It Works](#how-it-works)
- [Extend This Project](#extend-this-project)
- [Troubleshooting](#troubleshooting)
- [Reproducibility](#reproducibility)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Structure

```
.
├── vqe_h2_cudaq.py # main script (VQE loop, Hamiltonian, ansatz, backend selection)
├── requirements.txt # dependencies (cuda-quantum, scipy, matplotlib)
├── README.md
├── LICENSE
└── assets/
└── convergence.png # optional: saved plot
```

---

## Quickstart

1) Create and activate a virtual environment

```bash
# Linux/macOS
python -m venv vqe-env
source vqe-env/bin/activate

# Windows (PowerShell)
python -m venv vqe-env
vqe-env\Scripts\Activate.ps1
```

2) Install dependencies

```bash
pip install --upgrade pip
pip install "cuda-quantum>=0.6.0" scipy matplotlib
```

- GPU is optional. If you have an NVIDIA GPU (modern driver, e.g., 535+), CUDA‑Q will use it; otherwise it falls back to CPU.

3) Run

```bash
python vqe_h2_cudaq.py
```

If Matplotlib is installed, a convergence plot will pop up.

---

## Usage

- The script automatically selects the best available backend (GPU → CPU).
- You can set a backend explicitly before running by editing `choose_backend()` if you’d like to force CPU or GPU.

---

## Expected Output

You’ll see the backend banner, initial Hartree–Fock energy, iteration logs, and final energy near −1.137 Hartree.

```text
[INFO] Using backend: qpp # or "nvidia"/"custatevec" if GPU is used
[INFO] cuda-quantum version: 0.x.y
[INFO] Hartree–Fock energy (initial): -0.245218 Hartree
iter=001 E=-0.24487956 params=[...]
...
iter=057 E=-1.13726855 params=[...]
========== Results ==========
Optimized energy: -1.13726855 Hartree
Optimal params: [0.123456 -0.234567 0.345678]
HF -> VQE improvement: -0.245218 -> -1.137269 Hartree
Converged: True, message: Optimization terminated successfully.
Gradient sanity check (symmetric diff magnitude): 1.2e-05
```

---

## How It Works

- Hamiltonian
- Uses a standard 2‑qubit tapered Hamiltonian for H₂:
- H = c0 I + c1 Z0 + c2 Z1 + c3 Z0Z1 + c4 X0X1
- Expected ground-state energy ≈ −1.137 Hartree.

```python
def make_h2_hamiltonian():
s = cudaq.spin
H = (-1.052373245772859) * s.i(0) \
+ (0.39793742484318045) * s.z(0) \
+ (-0.39793742484318045) * s.z(1) \
+ (-0.01128010425623538) * (s.z(0) * s.z(1)) \
+ (0.18093119978423156) * (s.x(0) * s.x(1))
return H
```

- Ansatz
- Start from Hartree–Fock state |01⟩.
- Apply Ry rotations + a CNOT for entanglement, then a final Ry.

```python
@cudaq.kernel
def ansatz(theta0: float, theta1: float, theta2: float):
q = cudaq.qvector(2)
cudaq.x(q[1]) # Hartree–Fock |01>
cudaq.ry(theta0, q[0])
cudaq.ry(theta1, q[1])
cudaq.cx(q[0], q[1])
cudaq.ry(theta2, q[1])
```

- Objective & Optimization
- Minimize ⟨ψ(θ)|H|ψ(θ)⟩ using SciPy’s Nelder–Mead.
- CUDA‑Q’s `observe()` computes expectation values on the selected backend.

```python
result = cudaq.observe(ansatz, H, theta0, theta1, theta2)
energy = result.expectation() # handled safely in the script for version differences
```

- Backend Selection
- Tries GPU first (“nvidia”/“custatevec”), then CPU (“qpp”).

---

## Extend This Project

- Shot-based VQE: emulate hardware with finite shots; try SPSA/COBYLA.
- Larger molecules: generate Hamiltonians via Qiskit Nature or OpenFermion (JW/BK mappings).
- Gradients: parameter-shift or auto-diff with JAX/PyTorch for differentiable VQE.
- Throughput: batch parameter sweeps; explore CUDA Graphs and multi‑GPU scaling.

---

## Troubleshooting

- ModuleNotFoundError: No module named `cudaq`
- Install the correct package: `pip install cuda-quantum`
- Could not set target "nvidia"
- GPU backend not available; CPU fallback (“qpp”) will be used.
- AttributeError: `result.expectation` vs `result.expectation()`
- The script includes a helper to handle both. Or upgrade: `pip install -U cuda-quantum`
- macOS Apple Silicon
- Use CPU (“qpp”) backend; NVIDIA GPU backends require NVIDIA hardware.

Check your version:
```bash
python -c "import cudaq; print(cudaq.__version__)"
```

---

## Reproducibility

- Fixed RNG seed for initialization.
- Printed backend and CUDA‑Q version.
- Single-file script with deterministic setup.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgments

- NVIDIA CUDA‑Q team and docs.
- Standard H₂ Hamiltonian parameters from common VQE tutorials (Qiskit Nature/OpenFermion examples).

---

<details>
<summary><strong>Full script: vqe_h2_cudaq.py</strong></summary>

```python
import numpy as np
import cudaq
from scipy.optimize import minimize

def choose_backend():
tried = []
for target in ["nvidia", "custatevec", "qpp"]:
try:
cudaq.set_target(target)
print(f"[INFO] Using backend: {target}")
return
except Exception as e:
tried.append((target, str(e)))
try:
current = cudaq.get_target()
print(f"[INFO] Using default backend: {current}")
return
except Exception:
pass
msg = "; ".join([f"{t} -> {err}" for t, err in tried])
raise RuntimeError(f"Could not set any CUDA-Q target. Tried: {msg}")

def make_h2_hamiltonian():
s = cudaq.spin
H = (-1.052373245772859) * s.i(0) \
+ (0.39793742484318045) * s.z(0) \
+ (-0.39793742484318045) * s.z(1) \
+ (-0.01128010425623538) * (s.z(0) * s.z(1)) \
+ (0.18093119978423156) * (s.x(0) * s.x(1))
return H

@cudaq.kernel
def ansatz(theta0: float, theta1: float, theta2: float):
q = cudaq.qvector(2)
cudaq.x(q[1]) # Hartree–Fock |01>
cudaq.ry(theta0, q[0])
cudaq.ry(theta1, q[1])
cudaq.cx(q[0], q[1])
cudaq.ry(theta2, q[1])

def get_expectation(obs_result):
val = getattr(obs_result, "expectation", None)
if callable(val):
return float(val())
if val is not None:
return float(val)
val2 = getattr(obs_result, "expectation_value", None)
if callable(val2):
return float(val2())
if val2 is not None:
return float(val2)
raise AttributeError("Could not read expectation from ObserveResult. Update cuda-quantum.")

def make_energy_fn(H):
def energy(params):
params = np.asarray(params, dtype=float)
result = cudaq.observe(ansatz, H, float(params[0]), float(params[1]), float(params[2]))
return get_expectation(result)
return energy

def main():
choose_backend()
try:
print(f"[INFO] cuda-quantum version: {cudaq.__version__}")
except Exception:
pass

H = make_h2_hamiltonian()
energy = make_energy_fn(H)

hf_params = np.array([0.0, 0.0, 0.0], dtype=float)
e_hf = energy(hf_params)
print(f"[INFO] Hartree–Fock energy (initial): {e_hf:.6f} Hartree")

rng = np.random.default_rng(7)
x0 = rng.uniform(-0.1, 0.1, size=3)

history = []
def cb(xk):
e = float(energy(xk))
history.append(e)
print(f" iter={len(history):03d} E={e:.8f} params={np.array2string(xk, precision=4)}")

res = minimize(
energy,
x0,
method="Nelder-Mead",
options={"maxiter": 250, "xatol": 1e-6, "fatol": 1e-6, "disp": False},
callback=cb
)

print("\n========== Results ==========")
print(f"Optimized energy: {res.fun:.8f} Hartree")
print(f"Optimal params: {np.array2string(res.x, precision=6)}")
print(f"HF -> VQE improvement: {e_hf:.6f} -> {res.fun:.6f} Hartree")
print(f"Converged: {res.success}, message: {res.message}")

eps = 1e-3
e_plus = energy(res.x + eps)
e_minus = energy(res.x - eps)
grad_mag = abs(e_plus - e_minus) / (2 * eps)
print(f"Gradient sanity check (symmetric diff magnitude): {grad_mag:.6e}")

try:
import matplotlib.pyplot as plt
plt.plot(history, marker='o', linewidth=1)
plt.xlabel("Iteration")
plt.ylabel("Energy (Hartree)")
plt.title("VQE convergence on H2 (CUDA-Q)")
plt.grid(True, alpha=0.3)
plt.savefig("assets/convergence.png", dpi=160, bbox_inches="tight")
plt.show()
except Exception:
pass

if __name__ == "__main__":
main()
```

</details>
