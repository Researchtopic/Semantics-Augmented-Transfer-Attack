# Semantics-Augmented Transfer Attack

This is a pre-release version accompanying a manuscript currently under review. The code and data are being released progressively, and field names and directory layout may change. Author information and the paper link will be added once the manuscript is accepted.


### Dataset preparation

Download and extract the evaluation dataset and prompts into `dataset/` before running inference. The dataset download link will be added when the data are released.


### Dependencies

For LLaVA and BLIP inference, install the dependencies in a dedicated environment:

```bash
pip install -r requirements.txt
```

For MiniGPT-4, follow the separate repository and environment setup instructions at the top of `inference/minigpt4_inference.py`.