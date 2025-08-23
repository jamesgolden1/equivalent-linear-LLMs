# Equivalent Linear Mappings of Large Language Models

**A novel approach to interpreting transformer decoder models with equivalent linear reconstruction and decomposition.**

https://arxiv.org/abs/2505.24293

**James R. Golden**

## Key findings

We demonstrate that large language models can be mapped to nearly-exact equivalent linear systems for any given input sequence, without modifying model weights or altering predictions. We achieve this through strategic gradient computation modifications that create "detached Jacobians", which are linear representations that capture the complete forward computation.

### Why This Matters

- **Near-Exact Reconstruction**: The detached Jacobian linearly reconstructs the output embedding, where the subsequent token probabilities pass torch.allclose at 1-14
- **Interpretability**: Reveals semantic concepts emerging in model layers through the singular value decomposition
- **Efficiency**: Enables analysis of up to 14BBparameter models (Qwen 3 14B, Gemma 3 12 B, Llama 3.1 8B) passing torch.allclose at 1-14
- **Universality**: Works across model families (Qwen 3, Gemma 3, Llama 3, Phi 4, Mistral Ministral, OLMo 2)

## How It Works
### The Linear Path
Our approach exploits a fundamental structural property of transformer architectures wherein every operation (gated activations, attention, and normalization) can be expressed as $A(x) \cdot x$, where $A(x)$ represents an input-dependent coefficient matrix and $x$ preserves the linear pathway. To expose this linear structure, we strategically detach components of the gradient computation with respect to an input sequence, freezing the $A(x)$ terms at their values computed during inference. This ``detached’’ Jacobian of the model reconstructs the output with one linear operation per input token.  F

or example, $SiLU(x) = x \cdot sigmoid(x)$, but when the nonlinear $sigmoid(x)$ term is "frozen" for a specific input $x^\*$, the Jacobian computed numerically by torch autograd is linear in $x$ and exactly reconstructs $SiLU(x^*)$.
<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/jacobian_detached.png" width=45%/>
</p>

where the "detached Jacobian" **J**$^+(x^*)$ captures the full nonlinear computation as a linear system valid at input $x^\*$.

### Technical Approach

- **Normalization**: Detach variance computation from gradient path
- **Activations**: Freeze nonlinear terms in $SwiGLU/GELU/Swish$ functions
- **Attention**: Detach softmax operation while preserving linear $V$ multiplication
- **Analysis**: Apply SVD to understand learned representations and semantic emergence


<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/fig1-llama-detached-swiglu.png" width=75%/>
</p>

Fig. 1: The equivalent linear path through the $SwiGLU$ layer.

## Key Results
### Model Coverage

- Qwen 3 (8B - 32B parameters)
- Deepseek R1 0528 Qwen 3 (8B parameters)
- Gemma 3 (4B - 12B - 27B parameters)
- Llama 3 (3B - 8B - 70B parameters)
- Phi 4 (3B - 14B parameters)
- Mistral Ministral (8B parameters)
- OLMo 2 (8B parameters)

### Semantic Analysis

- **Low-rank structure**: Models operate in extremely low-dimensional subspaces
- **Concept emergence**: Semantic concepts appear in later transformer layers
- **Token relationships**: Singular vectors decode to semantically relevant input/output tokens
- **Steering applications**: Detached Jacobians enable efficient concept steering

### Example: "The bridge out of Marin is the"
Our analysis reveals:

- Top singular vectors decode to concepts like "Golden", "bridge", "highway"
- Layer-by-layer emergence of geographic and infrastructure concepts
- Extremely sparse activation patterns with few dominant features

### Usage 
Huggingface token with model access required. The code below runs on a [free colab T4 instance](https://github.com/jamesgolden1/llms-are-llms/blob/main/notebooks/run_detached_jacobian.ipynb).
```
import os
from google.colab import userdata

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = '1'
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')

os.system('git clone https://github.com/jamesgolden1/llms-are-llms.git')
os.chdir('llms-are-llms')
os.system('pip install -r requirements.txt --no-deps')
os.system(f'python -u run_detached_jacobian.py --hf_token {os.environ["HF_TOKEN"]} --model_name "llama-3.2-3b" --text "The Golden"')
```

## Applications
**Interpretability**

- Concept Analysis: Understand what drives model predictions
- Layer Dynamics: Track semantic emergence through transformer layers
- Feature Importance: Identify key input tokens and concepts for next-token prediction

<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/deepseek-R1-0528-qwen3-8b.png" width=100%/>
</p>

Fig 2: Results for Deepseek R1 0528 Qwen 3 8B.

**Model Steering**

- Efficient Control: Steer model outputs using detached Jacobians
- Concept Injection: Inject specific concepts (e.g., "Golden Gate Bridge") into continuations
- Safety Applications: Detect and potentially mitigate bias or toxic content

<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/steering.png" width=100%/>
</p>

Table 1: Steering results across models.

**Research Tools**

- Dimensionality Analysis: Measure effective dimensionality of learned representations
- Cross-model Comparisons: Compare semantic structures across model families
- Ablation Studies: Understand token contributions to output token prediction

## Detaching an MLP activation for an equivalent linear mapping

This code snippet shows how the Qwen 3 MLP has components frozen at inference to reveal its linear for a given input seequence. The output is the same as the original function. Only the gradient at inference is changed.

The detach() statement in the else clause makes the function linear.

```
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.training:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)).clone().detach() * self.up_proj(x))
        return down_proj
```

## License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments
This work builds on foundational research in:

- Transformer interpretability (Elhage et al., 2021)
- Locally linear ReLU neural networks (Mohan et al., 2019)
- Diffusion model linearity (Kadkhodaie et al., 2023)
