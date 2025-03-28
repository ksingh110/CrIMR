import streamlit as st

st.title("Hyperparameters Overview")
st.write("### Hyperparameters")

st.write("**Optimizer:** Adam")
st.write("**Learning Rate:** 0.0001")
st.write("**Clip Norm:** 1 (for controlling rate of gradient flow)")
st.write("**Batch Size:** 16")
st.write("**Epochs:** 50")
st.write("**Training Sample Size:** 10000 (5000 with CRY1 Mutations and 5000 Control)")
st.write("**Validation Sample Size:** 2000")
st.write("**Testing Sample Size:** 2000")
st.write("**Dropout:** 0.3 (30% of neurons off during training)")
st.write("**Learning Rate:** 0.001")
st.write("**Batch Size:** 32")
st.write("**Epochs:** 50")
