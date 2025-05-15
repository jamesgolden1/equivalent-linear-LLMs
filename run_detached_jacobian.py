import os
import gc
import torch
import argparse

def get_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", help="Hugging Face token (optional if set as environment variable)")
    parser.add_argument("--model_name", help="Examples: llama-3.2-3b, gemma-3-4b, qwen-3-8b")
    parser.add_argument("--text", help="Input text sequence, model predicts next token and analyzes the detached Jacobian")
    parser.add_argument("--run_all", help="run all analysis scripts it true, takes a long time; if false, run a few quick ones")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or args.hf_token
    if not token:
        print("No Hugging Face token found. Please either:")
        print("1. Set the HF_TOKEN environment variable:")
        print("   - Linux/Mac: export HF_TOKEN='your_token_here'")
        print("   - Windows: set HF_TOKEN=your_token_here")
        print("2. Or pass the token as an argument: --hf_token 'your_token_here'")
        exit(1)
    else:
        os.environ["HF_TOKEN"] = token

    model_name = args.model_name

    if model_name is None:
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
    else:
        if "llama" in model_name and "3.2" in model_name:
            model_name="meta-llama/Llama-3.2-3B-Instruct"
        elif "llama" in model_name and "3.1" in model_name:
            model_name="meta-llama/Llama-3.1-8B-Instruct"
        elif "llama" in model_name and "3.3" in model_name:
            model_name="unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
            print("A100 instance required")

        elif "gemma" in model_name and "4b" in model_name:
            model_name="google/gemma-3-4b-it"
        elif "gemma" in model_name and "27b" in model_name:
            model_name="unsloth/gemma-3-27b-it-bnb-4bit"

        elif "phi" in model_name and "mini" in model_name:
            model_name="microsoft/Phi-4-mini-reasoning"

        elif "phi" in model_name and "3.5" in model_name:
            model_name= "microsoft/Phi-3.5-mini-instruct"
        elif "phi" in model_name in model_name:
            model_name="microsoft/phi-4"

        elif "qwen" in model_name and "32" in model_name:
            model_name="unsloth/Qwen3-32B-bnb-4bit"
        elif "qwen" in model_name and "8" in model_name:
            model_name="Qwen/Qwen3-8B"

        elif "mistral" in model_name:
            model_name="mistralai/Ministral-8B-Instruct-2410"

        else:
            model_name="meta-llama/Llama-3.2-3B-Instruct"

    text = args.text 
    if text is None:
        text = "The bridge out of Marin is the"
    
    run_all = args.run_all
    if run_all is None:
        run_all = False
    
    return token, model_name, text, run_all

def main():
    token, model_name, text, run_all = get_inputs()

    from src.JacobianAnalyzer import JacobianAnalyzer as JacobianAnalyzer

    if "gemma" in model_name:
        from models.gemma_3.gemma_3_forward import model_forward
    elif "phi" in model_name:
        from models.phi_4.phi_4_forward import model_forward
    elif "qwen" in model_name:
        from models.qwen_3.qwen_3_forward import model_forward
    elif "mistral" in model_name:
        from models.mistral_ministral.mistral_ministral_forward import model_forward
    # elif "olmo" in model_name:
    #     from models.olmo_2.olmo_2_forward import model_forward
    # else: # llama_3_model model_forward already loaded by default
        # from models.llama_3.llama_3_forward import model_forward

    setattr(JacobianAnalyzer, 'model_forward', model_forward)
    # JacobianAnalyzer.model_forward = staticmethod(model_forward)

    model_name_short = '_'+model_name.split('/')[1]

    # Initialize the analyzer
    try:
        print("Loading", model_name)
        analyzer = JacobianAnalyzer(model_name=model_name)
    except ValueError:
        raise ValueError("Invalid model name. Acceptable names include 'llama-3.2', 'llama-3.3', 'gemma-3-4b', 'qwen-3-8b', 'phi-4', 'mistral-ministral', etc.")

    max_new_tokens=1
    temperature=1e-6

    # Generate output
    analyzer.generate(text, max_new_tokens, temperature);

    # # # Compute Jacobian
    analyzer.compute_jacobian()
    analyzer.compute_jacobian_nonlinear()
    analyzer.plot_jacobian_comparison(text,filename_png="fig3"+model_name_short)

    analyzer.compute_jacobian_row_col_norm(n_components=8)#, svs=1)
    analyzer.plot_singular_values(mode="row_col_vectors",filename_png="fig4_col"+model_name_short)

    analyzer.compute_jacobian_svd(n_components=24, svs=1)
    analyzer.plot_singular_values(filename_png="fig4"+model_name_short)

    analyzer.plot_jacobian_image(filename_png="fig2"+model_name_short)

    if run_all:
        num_layers = len(analyzer.model.model.layers)
        layerlist=list(range(1,num_layers,num_layers//8))
        analyzer.compute_jacobian_layers_svd(layerlist=layerlist,n_components=8,svs=8)
        analyzer.plot_singular_values(mode='singular_vectors_layers',key='layer',filename_png="fig5_svd_layers"+model_name_short)

        analyzer.compute_jacobian_layers_svd(layerlist=layerlist,n_components=8,svs=8,key='mlp')
        analyzer.plot_singular_values(mode='singular_vectors_layers',key='mlp',filename_png="fig5_svd_mlp"+model_name_short)

        analyzer.compute_jacobian_layers_svd(layerlist=layerlist,n_components=8,svs=8,key='attn')
        analyzer.plot_singular_values(mode='singular_vectors_layers',key='attn',filename_png="fig5_svd_attn"+model_name_short)

        analyzer.plot_path(filename_png="fig6_path")
        analyzer.plot_dimensionality(filename_png="fig6_dimensionality")

        analyzer.compute_jacobian_layers_svd(layerlist=layerlist,layer_mode='layerwise',n_components=8,svs=8)
        analyzer.plot_singular_values(mode='singular_vectors_layers_layerwise',key='layer',filename_png="fig5_svd_layers_layerwise"+model_name_short)

        analyzer.compute_jacobian_layers_svd(layerlist=layerlist,layer_mode='layerwise',n_components=8,svs=8,key='mlp')
        analyzer.plot_singular_values(mode='singular_vectors_layers_layerwise',key='mlp',filename_png="fig5_svd_mlp_layerwise"+model_name_short)

        analyzer.compute_jacobian_layers_svd(layerlist=layerlist,layer_mode='layerwise',n_components=8,svs=8,key='attn')
        analyzer.plot_singular_values(mode='singular_vectors_layers_layerwise',key='attn',filename_png="fig5_svd_attn_layerwise"+model_name_short)

        analyzer.plot_dimensionality(layerwise=True,filename_png="fig6_dimensionality_layerwise")

if __name__ == "__main__":
    main()