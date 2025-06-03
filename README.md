# Large Language Models are Locally Linear Mappings

**A novel approach to interpreting transformer decoder models with nearly-exact locally linear decompositions.**

https://arxiv.org/abs/2505.24293

**James R. Golden**

## Key findings

We demonstrate that large language models can be mapped to exactly equivalent linear systems for any given input sequence, without modifying model weights or altering predictions. We achieve this through strategic gradient computation modifications that create "detached Jacobians" - linear representations that capture the complete forward computation.

### Why This Matters

- **Near-Exact Reconstruction**: The detached Jacobian reconstructs outputs with ~10⁻⁶ relative error and $R^{2}$ > 0.99999
- **Interpretability**: Reveals semantic concepts emerging in model layers through SVD analysis
- **Efficiency**: Enables analysis of 70B+ parameter models without additional training
- **Universality**: Works across model families (Llama, Gemma, Qwen, Phi, Mistral, OLMo)

## How It Works
### The Core Innovation
We strategically detach gradients from nonlinear operations (activation functions, normalization, attention softmax) to create locally linear paths through the network. For example, SiLU(x) = x*sigmoid(x), but when the nonlinear sigmoid(x) term is "frozen" for a specific input x^\*, the Jacobian computed numerically by torch autograd is linear in x and exactly reconstructs SiLU(x^\*).
<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/jacobian_detached.png" width=45%/>
</p>
where J⁺ is the "detached Jacobian" that captures the full nonlinear computation as a linear system valid at input x*.

### Technical Approach

**Normalization**: Detach variance computation from gradient path
**Activations**: Freeze nonlinear terms in SwiGLU/GELU/Swish functions
**Attention**: Detach softmax operation while preserving linear V multiplication
**Analysis**: Apply SVD to understand learned representations and semantic emergence

## Key Results
### Model Coverage

✅ Llama 3 (3B - 70B parameters)

✅ Gemma 3 (4B - 12B parameters)

✅ Qwen 3 (8B parameters, including Deepseek R1)

✅ Deepseek R1 0528 Qwen 3 (8B parameters)

✅ Phi 4 (14B parameters)

✅ Mistral Ministral (8B parameters)

✅ OLMo 2 (8B)

### Semantic Analysis

**Low-rank structure**: Models operate in extremely low-dimensional subspaces
**Concept emergence**: Semantic concepts appear in later transformer layers
**Token relationships**: Singular vectors decode to semantically relevant input/output tokens
**Steering applications**: Detached Jacobians enable efficient concept steering

### Example: "The bridge out of Marin is the"
Our analysis reveals:

- Top singular vectors decode to concepts like "Golden", "bridge", "highway"
- Layer-by-layer emergence of geographic and infrastructure concepts
- Extremely sparse activation patterns with few dominant features

### Usage 
Huggingface token with model access required
```
import os
from google.colab import userdata
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
!git clone https://jamesgolden1:$GITHUB_TOKEN@github.com/jamesgolden1/llms-are-llms.git
cd llms-are-llms
!pip install -r requirements.txt --no-deps
!python run_detached_jacobian.py --hf_token $HF_TOKEN --model_name "llama-3.2-3b" --text "The Golden"
```

## Applications
**Interpretability**

Concept Analysis: Understand what drives model predictions
Layer Dynamics: Track semantic emergence through transformer layers
Feature Importance: Identify key input tokens and concepts for next-token prediction

<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/deepseek-R1-0528-qwen3-8b.png" width=100%/>
</p>

Fig 1: Results for Deepseek R1 0528 Qwen 3 8B.

**Model Steering**

Efficient Control: Steer model outputs using detached Jacobians
Concept Injection: Inject specific concepts (e.g., "Golden Gate Bridge") into continuations
Safety Applications: Detect and potentially mitigate bias or toxic content

<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/steering.png" width=100%/>
</p>

Table 1: Steering results across models.

**Research Tools**

Dimensionality Analysis: Measure effective dimensionality of learned representations
Cross-model Comparisons: Compare semantic structures across model families
Ablation Studies: Understand component contributions to predictions

## Citation
```
bibtex@article{golden2025llms,
  title={Large Language Models are Locally Linear Mappings},
  author={Golden, James R.},
  journal={arXiv preprint},
  year={2025}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This work builds on foundational research in:

Transformer interpretability (Elhage et al., 2021)

Locally linear neural networks (Mohan et al., 2019)

Diffusion model linearity (Kadkhodaie et al., 2023)
