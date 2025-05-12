# Large Language Models are Locally Linear Mappings
James Golden

# Abstract

We demonstrate that the inference operation of several open-weight large language models (LLMs) can be mapped to an exactly equivalent linear system for any input sequence without modifying the model weights or altering output predictions. 

Extending techniques from image diffusion models that exhibit local or piecewise linearity, we strategically alter the gradient computation with respect to a given input sequence such that the Jacobian of the LLM nearly exactly reproduces the forward prediction with an interpretable linear system. We demonstrate this approach across models (**Llama 3.3, Gemma 3, Qwen 3, Phi 4, Mistral Ministral and OLMo 2**) and show through the singular value decomposition of the Jacobian that these LLMs operate in extremely low-dimensional subspaces spanned by singular vectors that decode to interpretable semantic concepts. 

This approach also allows us to examine the operation of each successive layer (and its components) as exact linear systems and to directly manipulate predictions. Despite their expressive power and global nonlinearity, modern LLMs can be interpreted through exact linear decompositions that reveal their internal representations and computational mechanisms.

<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/fig1-llama-detached-swiglu.png"  width=0.8/>
</p>

Fig 1: The network components that are detached from the computational graph at inference to enable local linearity with respect to the input embedding vectors.

<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/fig3-jacobian-reconstruction.png"  width=0.7/>
</p>

Fig 2: The reconstruction error of the Jacobian of the original network compared to the reconstruction error of the detached Jacobian of the network with a modified gradient at inference (which produces the same outputs as the original network).

<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/fig4-llama-32-detached-svd.png" width=0.7/>
</p>

Fig 3: The right and left singular values of the Jacobian matrix corresponding to each input embedding vector can be decoded to tokens, demonstrating that the right singular vectors select for the input tokens (as expected), and the left singular vectors generate semantic concepts that appear in the decoded output.

Mathematically, the Taylor expansion of a nonlinear function like the transformer mapping of input embedding vectors to a predicted output embedding vector is:
<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/jacobian_taylor.png" width=0.4/>
</p>

By detaching the gradients of particular terms at inference with respect to the input embedding vectors, a linear path for the input is preserved through the transformer function such that the predicted output embedding is unchanged but the high order terms are all exactly zero. In other words, the network's inference (and all of its subcomponents) are locally linear for a particular inference sequence. The output can be nearly exactly reproduced by mutliplying the matrices of the detached Jacobian with the input embedding vectors. (The function is globally nonlinear, and the detached Jacobian must be computed numerically for each input sequence).

<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/jacobian_detached.png"  width=0.4/>
</p>

