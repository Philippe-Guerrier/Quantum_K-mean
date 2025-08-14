# Quantum K-Means (Qiskit) - Swap-Test Distance

Academic demo that implements **k-means clustering** using a **quantum-inspired distance** computed with a **swap test** in **Qiskit**. The notebook generates a 2D synthetic dataset, iteratively updates centroids with a quantum distance oracle, and visualizes the final clusters and centroids.

> Scope: 2 clusters (k=2), 2D features (for clarity), simulated on Qiskit Aer.

---

## Overview

- **Goal**: Explore how a quantum circuit (swap test) can act as a distance oracle inside a classical k-means loop.
- **Approach**: For each data point and centroid, build a short circuit that estimates a distance from measurement statistics and use it exactly like Euclidean distance in vanilla k-means.
- **Deliverable**: One Jupyter notebook `Quantum Project 1_plot.ipynb` that runs end-to-end on the QASM simulator and plots results.

---

## Method

### 1) Dataset
- Uses `sklearn.datasets.make_blobs(n_samples=300, centers=2, random_state=42)`.
- 2D points for easy plotting and for a small quantum register.

### 2) Distance via Swap Test
We encode information about the difference between a sample `x` and a centroid `y` into a 2–qubit circuit and use a **swap test** to estimate a similarity that we convert into a distance.

High-level steps for a single pair `(x, y)`:
1. Compute `diff = x - y`. If `‖diff‖ = 0`, distance is 0.
2. Normalize `diff` to get a 2D amplitude vector, and `initialize` it on qubit 0.
3. Apply swap-test pattern with qubit 1 as the ancilla: `H(1) → SWAP(0,1) → H(1) → measure(1)`.
4. Run on the simulator with `shots` (e.g., 2048) and compute `p0 = counts['0'] / shots`.
5. Define a distance, e.g., `d = sqrt(2 * (1 - p0))` (chosen for this demo).

> Note: The precise interpretation of the swap-test probability depends on the chosen encoding. Here we use a simple amplitude initialization of the **difference** for demonstration purposes.

### 3) K-Means Loop
- For each iteration:
  - Assign each point to the nearest centroid using the quantum distance.
  - Update centroids as the mean of assigned points.
  - Stop if the centroid shift is smaller than a tolerance (e.g., `1e-3`) or after `max_iterations`.

---

## Repository Structure

```
Quantum_K-mean/
├─ Quantum Project 1_plot.ipynb     # main notebook
└─ README.md                        # this file
```

---

## Requirements

- Python ≥ 3.9
- Qiskit (Aer + Terra)
- NumPy, scikit-learn, matplotlib, pandas

Recommended pinned set (works with the original notebook APIs):

```bash
pip install "qiskit==0.43.*" "qiskit-aer==0.12.*" numpy pandas scikit-learn matplotlib
```

If you prefer the latest Qiskit (≥ 1.0), APIs for execution changed. See the note at the end for an updated snippet.

---

## How to Run

### Option A - Jupyter
1. Create and activate a virtual environment (optional).
2. Install the requirements (see above).
3. Open the notebook and **Run All**.

### Option B - Convert to a Script (Optional)
You can extract the notebook logic into, e.g., `qkmeans.py`:
- Keep the functions `quantum_distance(...)` and `quantum_k_means(...)`.
- Use `matplotlib` to render the final scatter and centroids.

---

## Key Functions (from the notebook)

### Quantum Distance (execute-based API)
```python
def quantum_distance(x, y, backend):
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
    import numpy as np

    q = QuantumRegister(2)
    c = ClassicalRegister(1)
    circuit = QuantumCircuit(q, c)

    diff = x - y
    norm = np.linalg.norm(diff)
    if norm == 0:
        return 0.0
    diff_normalized = diff / norm

    circuit.initialize(diff_normalized, 0)
    circuit.h(1)
    circuit.swap(0, 1)
    circuit.h(1)
    circuit.measure(1, 0)

    result = execute(circuit, backend, shots=2048).result()
    counts = result.get_counts()
    p0 = counts.get("0", 0) / sum(counts.values())
    distance = np.sqrt(2 * (1 - p0))
    return distance
```

### K-Means Update
```python
def quantum_k_means(data_points, centroids, backend):
    import numpy as np

    new_centroids = np.zeros_like(centroids, dtype=float)
    counts = np.zeros(centroids.shape[0], dtype=int)

    for point in data_points:
        best_idx = None
        best_dist = float("inf")
        for idx, c in enumerate(centroids):
            d = quantum_distance(point, c, backend)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        new_centroids[best_idx] += point
        counts[best_idx] += 1

    counts[counts == 0] = 1
    new_centroids = new_centroids / counts[:, None]
    return new_centroids
```

---

## Example Workflow (Notebook)

```python
import numpy as np
from sklearn.datasets import make_blobs
from qiskit import Aer

# 1) Data
X, _ = make_blobs(n_samples=300, centers=2, random_state=42)

# 2) Init
centroids = np.array([[2, 2], [0, 0]], dtype=float)
backend = Aer.get_backend("qasm_simulator")
max_iterations = 10
tolerance = 1e-3

# 3) Iterate
for _ in range(max_iterations):
    new_centroids = quantum_k_means(X, centroids, backend)
    if np.linalg.norm(new_centroids - centroids) < tolerance:
        break
    centroids = new_centroids

print("Final centroids:", centroids)
```

Assignment and plot (as in the notebook):

```python
# Assign points
assignments = []
for p in X:
    best_idx = None
    best_dist = float("inf")
    for i, c in enumerate(centroids):
        d = quantum_distance(p, c, backend)
        if d < best_dist:
            best_dist = d
            best_idx = i
    assignments.append(best_idx)

# Visualize
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=assignments, cmap="viridis", s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x", s=100, label="Final Centroids")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Quantum K-Means Clustering (Swap-Test Distance)")
plt.show()
```

---

## Notes, Limitations, and Tips

- **Performance**: This demo calls a quantum circuit for each point–centroid pair at each iteration, which is computationally heavy for large datasets. Keep `n`, `k`, and `shots` small.
- **Shots**: More shots reduce variance in the estimated probability but slow down execution.
- **Encoding**: This uses a minimal 2-qubit scheme for differences. Other encodings (e.g., feature maps such as `ZZFeatureMap`) can be explored.
- **Versioning**: The original code used the legacy `execute` API. For Qiskit ≥ 1.0, use `backend.run(transpile(...))` (see below).

---

## Qiskit ≥ 1.0 Execution Snippet (If Needed)

If you install the latest Qiskit, replace the `execute(...)` call with:

```python
from qiskit_aer import AerSimulator
from qiskit import transpile

backend = AerSimulator()

def run_counts(circuit, backend, shots=2048):
    compiled = transpile(circuit, backend=backend)
    job = backend.run(compiled, shots=shots)
    result = job.result()
    return result.get_counts()

# Inside quantum_distance(...)
counts = run_counts(circuit, backend, shots=2048)
p0 = counts.get("0", 0) / sum(counts.values())
```

---

## Acknowledgements

- Qiskit (IBM Quantum) for quantum circuit simulation
- scikit-learn for data generation and scientific utilities
- Matplotlib for plotting
