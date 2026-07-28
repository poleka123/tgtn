# Tucker Dynamic Graph Neural ODE

This project implements a two-layer first-order Tucker-enhanced dynamic Graph
Neural ODE for multi-step spatio-temporal forecasting. It takes
`[batch, nodes, input_steps, features]` as input and returns
`[batch, nodes, output_steps, features]`.

The graph is derived from the continuously evolving hidden state during each
ODE evaluation; no pre-computed adjacency matrix is required. The model is in
`model/tucker_gode.py` and the training entry point is `main_sci_odegcn.py`.
