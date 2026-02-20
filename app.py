import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

# -------------------------------------------------
# PAGE SETUP
# -------------------------------------------------
st.set_page_config(page_title="Quantum Viterbi Decoder", layout="wide")
st.title("🧬 Quantum Convolutional Codes & Viterbi Decoding")

st.markdown("""
This app implements **ALL THREE PHASES**:
1. Quantum Convolutional Encoding  
2. State Diagram Construction  
3. Quantum-Inspired Viterbi Decoding  
""")

# =================================================
# PHASE 1 – QUANTUM CONVOLUTIONAL ENCODER
# =================================================
st.header("PHASE 1: Quantum Convolutional Encoding")

binary_msg = st.text_input("Enter binary message (e.g. 1011):")

def classical_convolutional_encoder(bits):
    """Rate 1/2, constraint length = 3"""
    mem1, mem2 = 0, 0
    output = []

    for b in bits:
        b = int(b)
        o1 = b ^ mem1 ^ mem2
        o2 = b ^ mem2
        output.append((o1, o2))
        mem2 = mem1
        mem1 = b

    return output

if st.button("Encode"):
    if not binary_msg or not set(binary_msg).issubset({"0", "1"}):
        st.error("Enter valid binary input")
    else:
        encoded = classical_convolutional_encoder(binary_msg)
        st.success("Encoding successful")

        st.write("### Encoded Output:")
        for i, out in enumerate(encoded):
            st.write(f"Input {binary_msg[i]} → Output {out}")

# =================================================
# PHASE 2 – STATE DIAGRAM
# =================================================
st.header("PHASE 2: State Diagram")

states = ["00", "01", "10", "11"]

def next_state(state, inp):
    mem1, mem2 = int(state[0]), int(state[1])
    new_state = f"{inp}{mem1}"
    o1 = inp ^ mem1 ^ mem2
    o2 = inp ^ mem2
    return new_state, (o1, o2)

selected_state = st.selectbox("Current State", states)
input_bit = st.selectbox("Input Bit", [0, 1])

if st.button("Get Next State"):
    ns, out = next_state(selected_state, input_bit)
    st.info(f"Next State: {ns}")
    st.info(f"Output: {out}")

# =================================================
# PHASE 3 – QUANTUM-INSPIRED VITERBI DECODING
# =================================================
st.header("PHASE 3: Quantum-Inspired Viterbi Decoding")

received = st.text_input(
    "Enter received bits (pairs, e.g. 110010):"
)

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

if st.button("Run Viterbi"):
    if len(received) % 2 != 0:
        st.error("Enter bits in pairs")
    else:
        received_pairs = [
            (int(received[i]), int(received[i+1]))
            for i in range(0, len(received), 2)
        ]

        path_metrics = {s: np.inf for s in states}
        path_metrics["00"] = 0
        paths = {s: "" for s in states}

        # FORWARD TRACE
        for r in received_pairs:
            new_metrics = {s: np.inf for s in states}
            new_paths = {s: "" for s in states}

            for s in states:
                if path_metrics[s] < np.inf:
                    for inp in [0, 1]:
                        ns, out = next_state(s, inp)
                        metric = path_metrics[s] + hamming(out, r)

                        if metric < new_metrics[ns]:
                            new_metrics[ns] = metric
                            new_paths[ns] = paths[s] + str(inp)

            path_metrics = new_metrics
            paths = new_paths

        st.subheader("Final Path Metrics")
        for s in states:
            st.write(f"State {s}: Metric = {path_metrics[s]}")

        # TRACEBACK
        best_state = min(path_metrics, key=path_metrics.get)
        decoded = paths[best_state]

        st.success(f"Decoded Message: {decoded}")
        st.info(f"Ending State: {best_state}")

# =================================================
# QUANTUM VISUALIZATION (OPTIONAL)
# =================================================
st.header("Quantum Path Superposition (Visualization)")

qc = QuantumCircuit(2)
qc.h([0, 1])
state = Statevector.from_instruction(qc)

st.write("Superposition of all paths:")
st.write(state)