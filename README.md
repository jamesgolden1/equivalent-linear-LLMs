# Large Language Models are Locally Linear Mappings
James Golden

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
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/deepseek-R1-0528-qwen3-8b.png" width=45%/>
</p>

Fig 4: Results for Deepseek R1 0528 Qwen 3 8B.

<p align="center">
  <img src="https://github.com/jamesgolden1/llms-are-llms/blob/main/images/jacobian_detached.png" width=45%/>
</p>

This approach therefore finds an exact linear system that reproduces the transformer's operation for a given sequence, and allows for the numerical intpretation of the LLM as a locally linear model. 
