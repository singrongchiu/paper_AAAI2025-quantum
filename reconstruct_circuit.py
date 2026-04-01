#!/usr/bin/env python3
"""
reconstruct_circuit.py
Reconstructs QCrank/VQDP QuantumCircuits from the inp_udata stored in .h5 files.

Usage:
    python reconstruct_circuit.py                       # uses default qc5adr_ibm_kingston.ibm.h5
    python reconstruct_circuit.py --h5 data/jobs/qc3adr_ibm_kingston.ibm.h5
    python reconstruct_circuit.py --h5 data/jobs/qc5adr_ibm_kingston.ibm.h5 --sample 0 --draw
    python reconstruct_circuit.py --h5 data/jobs/qc5adr_ibm_kingston.ibm.h5 --all --verify
"""

import argparse
import sys
import os
import numpy as np
from pprint import pprint

sys.path.insert(0, os.path.dirname(__file__))
from toolbox.Util_H5io4 import read4_data_hdf5

# angle encoding (matches values_to_angles in VQT code)
def values_to_angles(x: np.ndarray) -> np.ndarray:
    """arccos encoding: theta = arccos(x), x in [-1, 1]"""
    x = np.clip(x, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arccos(x)


# inline VQDPCircuitBuilder (no external VQT dependency needed)
def _apply_zero_control_fixes(qc, address_qubits, bitstring):
    for q, bit in zip(address_qubits, bitstring):
        if bit == "0":
            qc.x(q)

def _undo_zero_control_fixes(qc, address_qubits, bitstring):
    for q, bit in zip(address_qubits, bitstring):
        if bit == "0":
            qc.x(q)

def build_symbolic_circuit(num_address_qubits: int):
    """
    Builds the symbolic (parametric) QCrank/VQDP circuit.
    Returns (qc, alpha_params, beta_params).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    num_pairs = 2 ** num_address_qubits
    num_total_qubits = num_address_qubits + 2

    qc = QuantumCircuit(num_total_qubits, 1)
    address_qubits = list(range(num_address_qubits))
    x_qubit = num_address_qubits
    y_qubit = num_address_qubits + 1

    alpha = ParameterVector("alpha", num_pairs)
    beta  = ParameterVector("beta",  num_pairs)

    # QCrank: H on address qubits, then address-conditioned Ry rotations
    qc.h(address_qubits)

    for l in range(num_pairs):
        bitstring = format(l, f"0{num_address_qubits}b")
        _apply_zero_control_fixes(qc, address_qubits, bitstring)
        qc.mcry(alpha[l], q_controls=address_qubits, q_target=x_qubit,
                q_ancillae=[], mode="noancilla")
        qc.mcry(beta[l],  q_controls=address_qubits, q_target=y_qubit,
                q_ancillae=[], mode="noancilla")
        _undo_zero_control_fixes(qc, address_qubits, bitstring)

    # EHands block
    qc.barrier()
    qc.cx(x_qubit, y_qubit)
    qc.rz(np.pi / 2, y_qubit)

    # Measure second data qubit
    qc.measure(y_qubit, 0)

    return qc, alpha, beta


def reconstruct_circuit(inp_udata: np.ndarray, nq_addr: int, sample_idx: int):
    """
    Reconstruct one bound QuantumCircuit from inp_udata.

    inp_udata shape: (num_addr, 2, num_sample)
      axis 0 = address index l  (0 .. 2^nq_addr - 1)
      axis 1 = data channel     (0=x/alpha, 1=y/beta)
      axis 2 = sample index     (0 .. num_sample - 1)
    """
    from qiskit import QuantumCircuit

    num_pairs = 2 ** nq_addr
    assert inp_udata.shape[0] == num_pairs, \
        f"inp_udata axis-0 size {inp_udata.shape[0]} != 2^nq_addr={num_pairs}"

    x_vals = inp_udata[:, 0, sample_idx]   # raw values in [-1, 1]
    y_vals = inp_udata[:, 1, sample_idx]

    alpha_angles = values_to_angles(x_vals)  # arccos encoding
    beta_angles  = values_to_angles(y_vals)

    qc_sym, alpha_params, beta_params = build_symbolic_circuit(nq_addr)

    bind_map = {}
    for i in range(num_pairs):
        bind_map[alpha_params[i]] = float(alpha_angles[i])
        bind_map[beta_params[i]]  = float(beta_angles[i])

    return qc_sym.assign_parameters(bind_map)


def verify_against_metadata(qc, transpile_md: dict):
    """
    Cross-check the reconstructed circuit's stats against the h5 transpile metadata.
    Note: the metadata reflects the *transpiled* circuit; the reconstructed circuit
    is the logical (pre-transpile) version, so counts will differ.
    """
    ops = dict(qc.count_ops())
    print("\n-- Reconstructed circuit stats ----------------------")
    print(f"  Qubits      : {qc.num_qubits}")
    print(f"  Clbits      : {qc.num_clbits}")
    print(f"  Depth       : {qc.depth()}")
    print(f"  Gate counts : {ops}")

    print("\n-- H5 transpile metadata (transpiled circuit) -------")
    pprint(transpile_md)

    # Qubit count should match (logical == physical qubit count after transpile)
    match = qc.num_qubits == transpile_md.get('num_qubit')
    print(f"\n  num_qubit match : {'YES' if match else 'NO'} "
          f"(reconstructed={qc.num_qubits}, h5={transpile_md.get('num_qubit')})")
    return match


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Reconstruct QCrank circuits from .h5 files")
    parser.add_argument("--h5",     default="data/jobs/qc5adr_ibm_kingston.ibm.h5",
                        help="Path to the .ibm.h5 (or .ionq.h5) job file")
    parser.add_argument("--sample", type=int, default=0,
                        help="Which sample index to reconstruct (default: 0)")
    parser.add_argument("--all",    action="store_true",
                        help="Reconstruct all samples and print a summary")
    parser.add_argument("--draw",   action="store_true",
                        help="Print the circuit diagram (text)")
    parser.add_argument("--verify", action="store_true",
                        help="Cross-check reconstructed circuit against h5 metadata")
    args = parser.parse_args()

    # ── Load H5 ──────────────────────────────────────────────────────────────
    bigD, metaD = read4_data_hdf5(args.h5)
    pmd = metaD['payload']
    nq_addr    = pmd['nq_addr']
    num_sample = pmd['num_sample']
    inp_udata  = bigD['inp_udata']   # shape: (2^nq_addr, 2, num_sample)

    print(f"\nLoaded   : {args.h5}")
    print(f"nq_addr  : {nq_addr}  (address qubits)")
    print(f"num_addr : {2**nq_addr}  (= 2^nq_addr)")
    print(f"num_sample: {num_sample}")
    print(f"inp_udata : {inp_udata.shape}  (num_addr, 2=[x,y], num_sample)")

    if args.all:
        # Reconstruct every sample and summarise
        print(f"\nReconstructing all {num_sample} circuits...")
        circuits = []
        for s in range(num_sample):
            qc = reconstruct_circuit(inp_udata, nq_addr, s)
            circuits.append(qc)
        print(f"Done. {len(circuits)} circuits reconstructed.")
        print(f"Each has {circuits[0].num_qubits} qubits, depth {circuits[0].depth()} (pre-transpile).")
        return circuits

    # ── Single sample ─────────────────────────────────────────────────────────
    s = args.sample
    if s >= num_sample:
        print(f"ERROR: --sample {s} out of range (num_sample={num_sample})")
        sys.exit(1)

    print(f"\nReconstructing sample {s} ...")
    qc = reconstruct_circuit(inp_udata, nq_addr, s)
    print(f"Success. Circuit has {qc.num_qubits} qubits, depth {qc.depth()}.")

    if args.draw:
        print("\n── Circuit diagram ──────────────────────────────────")
        print(qc.draw(output='text', fold=120))

    if args.verify:
        verify_against_metadata(qc, metaD.get('transpile', {}))

    return qc


if __name__ == "__main__":
    main()
