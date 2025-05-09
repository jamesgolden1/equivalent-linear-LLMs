import torch 
def model_forward(self, embeds, lstart=0, lsplit=None, key='layer'):
    """
    Forward pass through Phi3 model layers

    Args:
        embeds: Input embeddings tensor of shape (batch_size, seq_length, hidden_size)
        lsplit (int): Number of layers to process (if None, process all layers)
        key (str): Return type - 'layer' for final layer output, 'attn' for attention output,
                  'mlp' for MLP output

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

    # Generate position embeddings
    position_embeddings = self.model.model.rotary_emb(embeds, position_ids)

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
            outdict['layer_input']=hidden_states
        elif key == 'attn_input':
            outdict['attn_input']=hidden_states

        # Store residual for skip connection
        residual = hidden_states

        # Apply input layer normalization
        hidden_states = self.model.model.layers[li].input_layernorm(hidden_states)

        # Apply self-attention
        if key == 'attn_heads':
            hidden_states, _, self_attn_weights_heads = self.model.model.layers[li].self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                cache_position=cache_position,
                return_heads=True
            )
        else:
            hidden_states, _ = self.model.model.layers[li].self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                cache_position=cache_position,
                return_heads=False
            )

        # Save attention output if requested
        if key == 'attn':
            if li == lsplit - 1:
                attn_output = self.model.model.norm(hidden_states)
                outdict['attn'] = attn_output
        elif key == 'attn_heads':
                # print("ATTN HEADS SHAPE:",self_attn_weights_heads.shape)
                # torch.Size([1, 7, 24, 128])
                sh=self_attn_weights_heads.shape
                attn_output_heads=self_attn_weights_heads
                attn_output_heads = self.model.model.norm(torch.reshape(self_attn_weights_heads,[sh[0],sh[1],sh[2]*sh[3]]))
                outdict['attn_heads'] = attn_output_heads.reshape(sh)
                # print("ATTN HEADS SHAPE:",outdict['attn_heads'].shape)

        # Add residual connection with dropout
        hidden_states = residual + self.model.model.layers[li].resid_attn_dropout(hidden_states)

        # Store residual for MLP
        residual = hidden_states

        if key == 'mlp_input':
            outdict['mlp_input']=hidden_states

        # Apply post attention layer normalization
        hidden_states = self.model.model.layers[li].post_attention_layernorm(hidden_states)

        # Apply MLP
        hidden_states = self.model.model.layers[li].mlp(hidden_states)
        # Save MLP output if requested
        if key == 'mlp':
            if li == lsplit - 1:
                mlp_output = self.model.model.norm(hidden_states)
                outdict['mlp'] = mlp_output

        # Add residual connection with dropout
        hidden_states = residual + self.model.model.layers[li].resid_mlp_dropout(hidden_states)

    # Apply final normalization if we processed all layers
    if key != "layer_input" and li == lsplit - 1:
        hidden_states = self.model.model.norm(hidden_states)

    # Store final layer output
    outdict['layer'] = hidden_states

    # Return the last token embedding of the first batch for the requested key
    if key == "layer_input" or key == "attn_input" or  key == "mlp_input":
        return outdict[key]
    else:
        return outdict[key][0, -1]