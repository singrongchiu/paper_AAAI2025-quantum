#!/bin/bash
# Reconstruct sample 0, verify metadata
# source .venv/bin/activate

python reconstruct_circuit.py --verify

# Reconstruct a different .h5 (e.g. 3 address qubits)
python reconstruct_circuit.py --h5 data/jobs/qc3adr_ibm_kingston.ibm.h5 --verify

# Reconstruct all 30 samples
python reconstruct_circuit.py --all

# Print circuit diagram (warning: very large for nq_addr=5)
python reconstruct_circuit.py --sample 2 --draw
