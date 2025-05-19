def model_forward(self, embeds, lstart=0, lsplit=None, key='layer'):
    """
    Forward pass through OLMo 2 model layers with linearization for Jacobian analysis

    Args:
        embeds: Input embeddings tensor of shape (batch_size, seq_length, hidden_size)
        lstart (int): First layer to process
        lsplit (int): Number of layers to process (if None, process all layers)
        key (str): Return type - 'layer' for final layer output, 'attn' for attention output, 
                  'mlp' for MLP output, 'layer_input', 'attn_input', or 'mlp_input' for inputs

    Returns:
        Hidden states tensor after processing the last token of the first batch
    """
    outdict = {}

    if lsplit is None:
        lsplit = len(self.model.model.layers)

    # Get batch size and sequence length
    batch_size, seq_length = embeds.shape[:2]

    # Create position IDs
    position_ids = torch.arange(seq_length, device=embeds.device).unsqueeze(0).expand(batch_size, -1)

    # Generate position embeddings for rotary attention
    cos, sin = self.model.model.rotary_emb(embeds, position_ids)
    position_embeddings = (cos, sin)

    # Create cache position for tracking position in sequence
    cache_position = torch.arange(seq_length, device=embeds.device)

    # Create basic attention mask (all ones)
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.float, device=embeds.device)

    # Generate causal mask
    causal_mask = self.model.model._update_causal_mask(
        attention_mask,
        embeds,
        cache_position,
        None,  # past_key_values
        False  # output_attentions
    )

    # Process through layers
    hidden_states = embeds

    for li in range(lstart, lsplit):
        if key == 'layer_input':
            outdict['layer_input'] = hidden_states
            
        # Store residual for skip connection
        residual = hidden_states

        # Self Attention
        attn_output, _ = self.model.model.layers[li].self_attn(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings
        )
        
        # Save attention output if requested
        if key == 'attn' and li == lsplit - 1:
            normalized_output = self.model.model.norm(attn_output)
            outdict['attn'] = normalized_output

        # Apply post-attention normalization (different from Llama)
        attn_output = self.model.model.layers[li].post_attention_layernorm(attn_output)
        
        # Add residual connection after normalization (key architectural difference from Llama)
        hidden_states = residual + attn_output

        if key == 'attn_input':
            outdict['attn_input'] = hidden_states
            
        # Store residual for MLP block
        residual = hidden_states

        # Apply MLP
        mlp_output = self.model.model.layers[li].mlp(hidden_states)
        
        if key == 'mlp_input':
            outdict['mlp_input'] = hidden_states
            
        # Save MLP output if requested
        if key == 'mlp' and li == lsplit - 1:
            normalized_output = self.model.model.norm(mlp_output)
            outdict['mlp'] = normalized_output
            
        # Apply post-feedforward normalization (different from Llama)
        mlp_output = self.model.model.layers[li].post_feedforward_layernorm(mlp_output)
        
        # Add residual connection after normalization
        hidden_states = residual + mlp_output
        
        # Save layer output if requested (after full layer processing)
        if key == 'layer' and li == lsplit - 1:
            normalized_output = self.model.model.norm(hidden_states)
            outdict['layer'] = normalized_output

    # If we haven't stored the output yet, store the final state
    if 'layer' not in outdict:
        if lsplit == len(self.model.model.layers):
            # Apply final normalization if we processed all layers
            hidden_states = self.model.model.norm(hidden_states)
        outdict['layer'] = hidden_states

    # Return the last token embedding of the first batch for the requested key
    if key in ["layer_input", "attn_input", "mlp_input"]:
        return outdict[key]
    else:
        return outdict[key][0, -1]