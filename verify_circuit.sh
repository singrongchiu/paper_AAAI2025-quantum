#!/bin/bash
# Verify reconstructed QCrank circuits against .h5 metadata
# Activate venv first: source .venv/bin/activate

set -e  # stop on first error

# ── 1. Quick sanity check (logical circuit stats, no transpilation) ───────────
echo "=== [1/4] Logical circuit stats (qc5adr) ==="
python reconstruct_circuit.py --verify

echo ""
echo "=== [2/4] Logical circuit stats (qc3adr) ==="
python reconstruct_circuit.py --h5 data/jobs/qc3adr_ibm_kingston.ibm.h5 --verify

# ── 2. Unitary equivalence: UCRYGate == naive mcry (use small circuit) ────────
# Uses Operator.equiv — scales as 2^(2*nqubits), so keep nq_addr small (3).
echo ""
echo "=== [3/4] Operator.equiv check (qc3adr, sample 0) ==="
python reconstruct_circuit.py --h5 data/jobs/qc3adr_ibm_kingston.ibm.h5 --equiv

# ── 3. Transpile comparison using FakeWashington + stored initial_layout ──────
# Uses the same 127-qubit Eagle r3 topology as ibm_kingston.
# Runs on qc5adr (the primary experiment); takes ~1 min.
echo ""
echo "=== [4/4] Transpile comparison on FakeWashingtonV2 (qc5adr) ==="
python reconstruct_circuit.py --transpile

echo ""
echo "Done."
