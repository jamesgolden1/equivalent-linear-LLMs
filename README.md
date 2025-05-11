# Large Language Models are Locally Linear Mappings
James Golden

# Abstract

We demonstrate that the inference operation of several open-weight large language models (LLMs) can be mapped to an exactly equivalent linear system for any input sequence without modifying the model weights or altering output predictions. 

Extending techniques from image diffusion models that exhibit local or piecewise linearity, we strategically alter the gradient computation with respect to a given input sequence such that the Jacobian of the LLM nearly exactly reproduces the forward prediction with an interpretable linear system. We demonstrate this approach across models (**Llama 3.3, Gemma 3, Qwen 3, Phi 4, Mistral Ministral and OLMo 2**) and show through the singular value decomposition of the Jacobian that these LLMs operate in extremely low-dimensional subspaces spanned by singular vectors that decode to interpretable semantic concepts. 

This approach also allows us to examine the operation of each successive layer (and its components) as exact linear systems and to directly manipulate predictions. Despite their expressive power and global nonlinearity, modern LLMs can be interpreted through exact linear decompositions that reveal their internal representations and computational mechanisms.

![alt text](https://github.com/jamesgolden1/llms-are-llms/blob/main/images/fig1-llama-detached-swiglu.png "Llama 3 Detached Jacobian Architecture")
Fig 1: The network components that are detached from the computational graph at inference to enable local linearity with respect to the input embedding vectors.

![alt text](https://github.com/jamesgolden1/llms-are-llms/blob/main/images/fig3-jacobian-reconstruction.png "Llama 3.2 Detached Jacobian Reconstruction Error")
Fig 2: The reconstruction error of the Jacobian of the original network compared to the reconstruction error of the detached Jacobian of the network with a modified gradient at inference (which produces the same outputs as the original network).

