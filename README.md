# Large Language Models are Locally Linear Mappings
## A novel approach to interpreting transformer decoder models with nearly-exact locally linear decompositions

James R. Golden
https://arxiv.org/abs/2505.24293

## Key findings

We demonstrate that large language models can be mapped to exactly equivalent linear systems for any given input sequence, without modifying model weights or altering predictions. We achieve this through strategic gradient computation modifications that create "detached Jacobians" - linear representations that capture the complete forward computation.

## Why This Matters

- **Near-Exact Reconstruction**: Linear systems reproduce model outputs with ~10⁻⁶ relative error and R^2 > 0.99999
- **Interpretability**: Reveals semantic concepts emerging in model layers through SVD analysis
- **Efficiency**: Enables analysis of 70B+ parameter models without additional training
- **Universality**: Works across model families (Llama, Gemma, Qwen, Phi, Mistral, OLMo)

## How It Works
The Core Innovation
We strategically detach gradients from nonlinear operations (activation functions, normalization, attention softmax) to create locally linear paths through the network. For example, SiLU(x) = x*sigmoid(x), but when the nonlinear sigmoid(x) term is "frozen" for a specific input x^\*, the Jacobian computed numerically by torch autograd is linear in x and exactly reconstructs SiLU(x^\*).
<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/jacobian_detached.png" width=45%/>
</p>
where J⁺ is the "detached Jacobian" that captures the full nonlinear computation as a linear system valid at input x*.

## Technical Approach

**Normalization**: Detach variance computation from gradient path
**Activations**: Freeze nonlinear terms in SwiGLU/GELU/Swish functions
**Attention**: Detach softmax operation while preserving linear V multiplication
**Analysis**: Apply SVD to understand learned representations and semantic emergence

## Model Coverage

✅ Llama 3 (3B - 70B parameters)
✅ Gemma 3 (4B - 12B parameters)
✅ Qwen 3 (8B parameters, including Deepseek R1)
✅ Deepseek R1 0528 Qwen 3 (8B parameters)
✅ Phi 4 (14B parameters)
✅ Mistral Ministral (8B parameters)
✅ OLMo 2 (8B)

## Semantic Analysis

**Low-rank structure**: Models operate in extremely low-dimensional subspaces
**Concept emergence**: Semantic concepts appear in later transformer layers
**Token relationships**: Singular vectors decode to semantically relevant input/output tokens
**Steering applications**: Detached Jacobians enable efficient concept steering

## Example: "The bridge out of Marin is the"
Our analysis reveals:

- Top singular vectors decode to concepts like "Golden", "bridge", "highway"
- Layer-by-layer emergence of geographic and infrastructure concepts
- Extremely sparse activation patterns with few dominant features

## Usage 
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
Feature Importance: Identify key input features for any prediction

**Model Steering**

Efficient Control: Steer model outputs using detached Jacobians
Concept Injection: Inject specific concepts (e.g., "Golden Gate Bridge") into continuations
Safety Applications: Detect and potentially mitigate bias or toxic content

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

# Abstract

We demonstrate that the inference operations of several open-weight large language models (LLMs) can be mapped to an exactly equivalent linear system for an input sequence without modifying the model weights or altering output predictions. Extending techniques from image diffusion models that exhibit local or piecewise linearity, we strategically alter the gradient computation with respect to a given input sequence for a next-token prediction such that the Jacobian of the model nearly exactly reproduces the forward prediction with a linear system. 

We demonstrate this approach across models (Llama 3, Gemma 3, Qwen 3, Phi 4, Mistral Ministral and OLMo 2, up to Llama 3.3 70B Q4) and show through the singular value decomposition of the detached Jacobian that these LLMs operate in extremely low-dimensional subspaces where many of the largest singular vectors decode to concepts related to the most-likely output token. This approach also allows us to examine the operation of each successive layer (and its attention and MLP components) as nearly-exact linear systems and observe the emergence of semantic concepts. 

Despite their expressive power and global nonlinearity, modern LLMs can be interpreted through nearly-exact locally linear decompositions that provide insights into their internal representations and reveal interpretable semantic structures in the next-token prediction process.
<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/fig1-llama-detached-swiglu.png" width=70%/>
</p>

Fig 1: The Llama 3 architecture with gradient detachments that produce local linearity. The network components that are detached from the computational graph at inference to enable local linearity with respect to the input embedding vectors are outlined in red. These include normalization layers, feed forward/multi-layer perceptron blocks and attention blocks. This approach only works for LLMs with gated linear activations (Swish/SiLU, SwiGLU, GeLU) and zero-bias linear layers (for both feedfoward/MLP and attention blocks).

<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/fig3-jacobian-reconstruction.png" width=50%"/>
</p>

Fig 2: The reconstruction error of the Jacobian of the original network compared to the reconstruction error of the detached Jacobian of the network with a modified gradient at inference (which produces the same outputs as the original network).

<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/fig4-llama-32-detached-svd.png" width=70%"/>
</p>

Fig 3: The right and left singular values of the Jacobian matrix corresponding to each input embedding vector can be decoded to tokens, demonstrating that the right singular vectors select for the input tokens (as expected), and the left singular vectors generate semantic concepts that appear in the decoded output.

<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/deepseek-R1-0528-qwen3-8b.png" width=100%/>
</p>

Fig 4: Results for Deepseek R1 0528 Qwen 3 8B.

<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/jacobian_detached.png" width=45%/>
</p>

This approach therefore finds an exact linear system that reproduces the transformer's operation for a given sequence, and allows for the numerical intpretation of the LLM as a locally linear model. 
