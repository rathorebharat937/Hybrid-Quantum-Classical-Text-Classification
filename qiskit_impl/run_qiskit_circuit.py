"""
Run and Visualize Qiskit Quantum Circuit
=======================================
Executes the Qiskit quantum circuit using statevector simulation,
profiles circuit gate counts, and renders circuit diagrams.
"""

import os
import sys
import json
import time
import numpy as np

# Ensure root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from qiskit_impl.qiskit_circuit import build_qiskit_circuit, compute_pauli_z_expectations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs(os.path.join(ROOT_DIR, "results/qiskit/figures"), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "results/qiskit/logs"), exist_ok=True)

def run_simulation_demo():
    print("=" * 70)
    print("EXECUTING QISKIT QUANTUM CIRCUIT SIMULATION DEMONSTRATION")
    print("=" * 70)

    n_qubits = 4
    n_layers = 2

    # Deterministic test inputs representing 4-dimensional compressed text features
    inputs = np.array([0.452, -0.718, 0.284, -0.639], dtype=np.float32)

    # Initialize reproducible weights
    np.random.seed(42)
    weights = np.random.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, 3)).astype(np.float32)

    t0 = time.time()
    qc = build_qiskit_circuit(inputs, weights, n_qubits=n_qubits, n_layers=n_layers)
    circuit_build_time = time.time() - t0

    t1 = time.time()
    exp_vals = compute_pauli_z_expectations(qc, n_qubits=n_qubits)
    sim_time = time.time() - t1

    # Gate analysis
    ops_dict = qc.count_ops()
    total_gates = sum(ops_dict.values())
    depth = qc.depth()
    total_params = weights.size

    print(f"Number of Qubits:           {n_qubits}")
    print(f"Number of Variational Layers:{n_layers}")
    print(f"Circuit Depth:              {depth}")
    print(f"Total Quantum Operations:   {total_gates}")
    print(f"Gate Decomposition:         {dict(ops_dict)}")
    print(f"Trainable Quantum Weights:  {total_params} angles")
    print(f"Feature Encoding Vector:    {inputs.tolist()}")
    print(f"Output <Z_i> Expectations:  {exp_vals.tolist()}")
    print(f"Statevector Simulation Time:{sim_time * 1000:.3f} ms")

    # Generate ASCII diagram and save text representation
    text_diagram = qc.draw(output='text')
    text_circuit_path = os.path.join(ROOT_DIR, "results/qiskit/circuit_diagram.txt")
    with open(text_circuit_path, "w", encoding="utf-8") as f:
        f.write(str(text_diagram))
    print(f"\nSaved text circuit diagram to: results/qiskit/circuit_diagram.txt")

    # Save PNG circuit diagram
    fig_path = os.path.join(ROOT_DIR, "results/qiskit/figures/qiskit_quantum_circuit.png")
    try:
        # Use matplotlib drawer
        fig = qc.draw(output='mpl', style='iqp')
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved circuit figure to:       results/qiskit/figures/qiskit_quantum_circuit.png")
    except Exception as e:
        print(f"Warning: mpl drawer fallback: {e}")
        # Text-based matplotlib fallback if latex/mpl style issues arise
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.01, 0.99, str(text_diagram), family='monospace', fontsize=8, va='top')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()

    # Save summary metadata JSON
    exec_metadata = {
        "framework": "Qiskit 2.6.2",
        "simulator_backend": "Statevector (exact analytical)",
        "n_qubits": int(n_qubits),
        "n_layers": int(n_layers),
        "circuit_depth": int(depth),
        "total_gates": int(total_gates),
        "gate_counts": {k: int(v) for k, v in ops_dict.items()},
        "trainable_parameters": int(total_params),
        "input_features": inputs.tolist(),
        "pauli_z_expectations": [float(x) for x in exp_vals],
        "circuit_build_time_ms": float(circuit_build_time * 1000),
        "simulation_time_ms": float(sim_time * 1000)
    }

    json_path = os.path.join(ROOT_DIR, "results/qiskit/circuit_execution.json")
    with open(json_path, "w") as f:
        json.dump(exec_metadata, f, indent=2)
    print(f"Saved execution metadata to:   results/qiskit/circuit_execution.json")

    # Save detailed log
    log_path = os.path.join(ROOT_DIR, "results/qiskit/circuit_execution.log")
    with open(log_path, "w") as f:
        f.write("=== QISKIT QUANTUM CIRCUIT EXECUTION LOG ===\n")
        f.write(json.dumps(exec_metadata, indent=2) + "\n\n")
        f.write("CIRCUIT REPRESENTATION:\n")
        f.write(str(text_diagram) + "\n")
    print(f"Saved execution log to:        results/qiskit/circuit_execution.log")
    print("=" * 70)

if __name__ == "__main__":
    run_simulation_demo()

