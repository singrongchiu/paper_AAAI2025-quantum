#!/usr/bin/env python3
"""
reconstruct_circuit.py
Reconstructs QCrank/VQDP QuantumCircuits from the inp_udata stored in .h5 files.

Usage:
    python reconstruct_circuit.py                          # reconstruct sample 0 from default h5
    python reconstruct_circuit.py --h5 data/jobs/qc3adr_ibm_kingston.ibm.h5
    python reconstruct_circuit.py --sample 0 --draw        # print circuit diagram
    python reconstruct_circuit.py --sample 0 --verify      # compare qubit count vs h5 metadata
    python reconstruct_circuit.py --sample 0 --transpile   # compile + compare gate counts to h5
    python reconstruct_circuit.py --all                    # reconstruct all samples
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


# UCRYGate: uniformly-controlled Ry — the correct QCrank decomposition.
# For n address qubits, UCRYGate encodes 2^n angles using 2*(2^n) CX gates
# (linear in addresses), vs naive mcry which uses ~40*2^n CX gates.
# NOTE: UCRYGate requires concrete float angles (not symbolic Parameters),
#       so we build the bound circuit directly.
def build_bound_circuit(alpha_angles: np.ndarray, beta_angles: np.ndarray,
                        num_address_qubits: int):
    """
    Builds a fully-bound QCrank/VQDP circuit using UCRYGate.
    UCRYGate requires concrete float angles — no symbolic parameters.

    alpha_angles, beta_angles: arrays of length 2^num_address_qubits
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UCRYGate

    num_pairs = 2 ** num_address_qubits
    assert len(alpha_angles) == num_pairs
    assert len(beta_angles)  == num_pairs

    num_total_qubits = num_address_qubits + 2
    qc = QuantumCircuit(num_total_qubits, 1)

    address_qubits = list(range(num_address_qubits))
    x_qubit = num_address_qubits
    y_qubit = num_address_qubits + 1

    # QCrank: H on address qubits, then uniformly controlled Ry on data qubits.
    # UCRYGate qubit order is [target] + controls, with controls in LSB-first
    # (little-endian) order — so we reverse address_qubits.
    qc.h(address_qubits)
    qc.append(UCRYGate([float(a) for a in alpha_angles]), [x_qubit] + list(reversed(address_qubits)))
    qc.append(UCRYGate([float(b) for b in beta_angles]),  [y_qubit] + list(reversed(address_qubits)))

    # EHands block
    qc.barrier()
    qc.cx(x_qubit, y_qubit)
    qc.rz(np.pi / 2, y_qubit)

    # Measure second data qubit
    qc.measure(y_qubit, 0)

    return qc


def reconstruct_circuit(inp_udata: np.ndarray, nq_addr: int, sample_idx: int):
    """
    Reconstruct one bound QuantumCircuit from inp_udata.

    inp_udata shape: (num_addr, 2, num_sample)
      axis 0 = address index l  (0 .. 2^nq_addr - 1)
      axis 1 = data channel     (0=x/alpha, 1=y/beta)
      axis 2 = sample index     (0 .. num_sample - 1)
    """
    num_pairs = 2 ** nq_addr
    assert inp_udata.shape[0] == num_pairs, \
        f"inp_udata axis-0 size {inp_udata.shape[0]} != 2^nq_addr={num_pairs}"

    x_vals = inp_udata[:, 0, sample_idx]   # raw values in [-1, 1]
    y_vals = inp_udata[:, 1, sample_idx]

    alpha_angles = values_to_angles(x_vals)  # arccos encoding
    beta_angles  = values_to_angles(y_vals)

    return build_bound_circuit(alpha_angles, beta_angles, nq_addr)


def verify_against_metadata(qc, transpile_md: dict):
    """
    Compare the LOGICAL (pre-transpile) circuit to h5 metadata.
    Gate counts will differ because the h5 stores post-transpile stats.
    Only num_qubit should match directly. Use --transpile for a true comparison.
    """
    ops = dict(qc.count_ops())
    print("\n-- Reconstructed circuit (logical, pre-transpile) ---")
    print(f"  Qubits      : {qc.num_qubits}")
    print(f"  Clbits      : {qc.num_clbits}")
    print(f"  Depth       : {qc.depth()}")
    print(f"  Gate counts : {ops}")
    print("  NOTE: gate counts here are NOT comparable to h5 -- run --transpile for that.")

    print("\n-- H5 transpile metadata (post-transpile, ran on HW) ----")
    pprint(transpile_md)

    match = qc.num_qubits == transpile_md.get('num_qubit')
    print(f"\n  num_qubit match : {'YES' if match else 'NO'} "
          f"(reconstructed={qc.num_qubits}, h5={transpile_md.get('num_qubit')})")
    return match


def transpile_and_compare(qc, transpile_md: dict):
    """
    Transpile the reconstructed circuit using FakeWashingtonV2 (127-qubit Eagle r3
    processor — same topology as ibm_kingston) with the exact initial_layout stored
    in the h5 metadata, at optimization_level=3.
    This is the closest achievable apples-to-apples comparison.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2

    # FakeWashingtonV2: 127-qubit Eagle r3, same chip topology as ibm_kingston
    backend = FakeWashingtonV2()
    phys_qubits = transpile_md.get('phys_qubits')  # e.g. [83, 82, 105, 102, 104, 103, 96]

    print(f"\nTranspiling on FakeWashingtonV2 (127-qubit Eagle r3, ~= ibm_kingston)")
    print(f"  initial_layout = {phys_qubits}  (from h5 metadata)")
    print("  optimization_level = 3")
    print("  (This may take a minute)")

    pm = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        initial_layout=phys_qubits
    )
    qc_t = pm.run(qc)

    # Count 2q gates in the transpiled circuit
    ops_t = dict(qc_t.count_ops())
    two_q_names = {'cx', 'ecr', 'cz', 'swap', 'rzz', 'rxx', 'ryy'}
    n2q = sum(v for k, v in ops_t.items() if k in two_q_names)
    depth_2q = qc_t.depth(filter_function=lambda x: x.operation.num_qubits == 2)

    print("\n-- Transpiled circuit (FakeWashington, opt=3) --------")
    print(f"  Depth (total)   : {qc_t.depth()}")
    print(f"  2q gate depth   : {depth_2q}")
    print(f"  2q gate count   : {n2q}")
    print(f"  Gate counts     : {ops_t}")

    print("\n-- H5 transpile metadata (ran on ibm_kingston) ------")
    pprint(transpile_md)

    h5_2q_count = transpile_md.get('2q_gate_count', '?')
    h5_2q_depth = transpile_md.get('2q_gate_depth', '?')
    print("\n-- Comparison ---")
    print(f"  2q gate count : reconstructed={n2q}  h5={h5_2q_count}")
    print(f"  2q gate depth : reconstructed={depth_2q}  h5={h5_2q_depth}")
    print()
    print("  Note: random_compilation=True on the original run adds extra gates;")
    print("  exact equality is not expected even with the same topology.")
    return qc_t


def equiv_check(inp_udata: np.ndarray, nq_addr: int, sample_idx: int):
    """
    Build the same circuit using both UCRYGate and naive mcry, then compare
    their unitaries with Operator.equiv (checks equality up to global phase).
    Proves the two decompositions implement the same quantum operation.

    NOTE: measurements are removed before computing the operator, since
    Operator only supports unitary (measurement-free) circuits.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import UCRYGate

    num_pairs = 2 ** nq_addr
    x_vals = inp_udata[:, 0, sample_idx]
    y_vals = inp_udata[:, 1, sample_idx]
    alpha  = values_to_angles(x_vals)
    beta   = values_to_angles(y_vals)

    # ── Circuit A: UCRYGate (our reconstruction) ──────────────────────────────
    qc_ucry = build_bound_circuit(alpha, beta, nq_addr)

    # ── Circuit B: naive multi-controlled Ry ─────────────────────────────────
    num_total = nq_addr + 2
    qc_mcry = QuantumCircuit(num_total)
    addr = list(range(nq_addr))
    xq, yq = nq_addr, nq_addr + 1
    qc_mcry.h(addr)
    for l in range(num_pairs):
        bits = format(l, f"0{nq_addr}b")
        for q, b in zip(addr, bits):
            if b == "0": qc_mcry.x(q)
        qc_mcry.mcry(float(alpha[l]), q_controls=addr, q_target=xq,
                     q_ancillae=[], mode="noancilla")
        qc_mcry.mcry(float(beta[l]),  q_controls=addr, q_target=yq,
                     q_ancillae=[], mode="noancilla")
        for q, b in zip(addr, bits):
            if b == "0": qc_mcry.x(q)
    qc_mcry.cx(xq, yq)
    qc_mcry.rz(np.pi / 2, yq)

    # Strip measurements (Operator requires unitary circuits)
    def remove_meas(qc):
        return qc.copy().remove_final_measurements(inplace=False)

    qc_a = remove_meas(qc_ucry)
    qc_b = qc_mcry  # mcry version has no measurements

    print(f"\nComputing Operator for UCRYGate circuit  ({qc_a.num_qubits} qubits, 2^n={2**qc_a.num_qubits} dim)...")
    op_a = Operator(qc_a)
    print("Computing Operator for naive mcry circuit...")
    op_b = Operator(qc_b)

    result = op_a.equiv(op_b)
    print(f"\nOperator.equiv(UCRYGate, mcry) = {result}")
    if result:
        print("  The two circuits implement the SAME unitary (up to global phase).")
        print("  The UCRYGate reconstruction is correct.")
    else:
        print("  WARNING: unitaries differ — circuits are NOT equivalent.")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Reconstruct QCrank circuits from .h5 files")
    parser.add_argument("--h5",       default="data/jobs/qc5adr_ibm_kingston.ibm.h5",
                        help="Path to the .ibm.h5 (or .ionq.h5) job file")
    parser.add_argument("--sample",   type=int, default=0,
                        help="Which sample index to reconstruct (default: 0)")
    parser.add_argument("--all",      action="store_true",
                        help="Reconstruct all samples and print a summary")
    parser.add_argument("--draw",     action="store_true",
                        help="Print the circuit diagram (text)")
    parser.add_argument("--verify",    action="store_true",
                        help="Show logical circuit stats vs h5 metadata (pre-transpile)")
    parser.add_argument("--transpile", action="store_true",
                        help="Transpile on FakeWashington w/ stored initial_layout, compare to h5 (slow)")
    parser.add_argument("--equiv",     action="store_true",
                        help="Prove UCRYGate == naive mcry via Operator.equiv (slow for large nq_addr)")
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
        print("\n-- Circuit diagram --")
        print(qc.draw(output='text', fold=120))

    if args.verify:
        verify_against_metadata(qc, metaD.get('transpile', {}))

    if args.transpile:
        transpile_and_compare(qc, metaD.get('transpile', {}))

    if args.equiv:
        equiv_check(inp_udata, nq_addr, s)

    return qc


if __name__ == "__main__":
    main()
