# Configuration

`configs/default.yaml` is the canonical runtime configuration. The model's
architecture and feature contract are separately stored beside the checkpoint in
`weights/dgcnn_best/model.yaml`; runtime values may not silently override that
contract.

Parameters used for final test evaluation must be selected on training or
validation data and then frozen.

