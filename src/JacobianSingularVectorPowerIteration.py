#####@title Jacobian k svd (slow)
import torch
import torch.nn.functional as F
from torch.func import jvp, vjp
import numpy as np
import math
from contextlib import contextmanager


@contextmanager
def disable_flash_attention():
    """Context manager to temporarily disable Flash Attention for JVP/VJP compatibility."""
    # Store original states
    original_flash = torch.backends.cuda.flash_sdp_enabled()
    original_mem_efficient = torch.backends.cuda.mem_efficient_sdp_enabled()
    original_math = torch.backends.cuda.math_sdp_enabled()

    try:
        # Disable Flash Attention and Memory Efficient Attention, enable Math Attention
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        yield
    finally:
        # Restore original states
        torch.backends.cuda.enable_flash_sdp(original_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(original_mem_efficient)
        torch.backends.cuda.enable_math_sdp(original_math)


def randomized_svd_jacobian_improved(func, inputs, num_singular_vectors=5, num_iter=4,
                                   oversampling=10, debug=False, stabilize=True):
    """
    Improved randomized SVD for Jacobian matrices using JVP/VJP operations.

    Key improvements:
    1. Consistent flattening approach that matches torch.autograd.functional.jacobian
    2. Proper matrix B formation following Halko et al. algorithm exactly
    3. Better numerical stability and error handling
    4. Cleaner code structure with better separation of concerns

    Args:
        func: Function whose Jacobian we want to analyze
        inputs: Input tensor(s) to the function
        num_singular_vectors: Number of top singular vectors to compute
        num_iter: Number of power iterations for accuracy
        oversampling: Extra random vectors for stability
        debug: Print debugging information
        stabilize: Use numerical stabilization techniques

    Returns:
        U: Left singular vectors (output space)
        S: Singular values
        V: Right singular vectors (input space)
    """

    # Setup and dimension calculation
    device, dtype, input_dim, output_dim, input_shape, output_shape = _setup_dimensions(inputs, func, debug)

    k = num_singular_vectors

    # Adaptive parameters for small problems
    if min(input_dim, output_dim) < 20:
        # For very small problems, use more aggressive settings
        oversampling = max(oversampling, min(input_dim, output_dim) - k)
        num_iter = max(num_iter, 6)
        if debug:
            print(f"Small problem detected: using oversampling={oversampling}, num_iter={num_iter}")

    l = min(k + oversampling, min(input_dim, output_dim))

    if debug:
        print(f"Input dim: {input_dim}, Output dim: {output_dim}, k={k}, l={l}")

    # Create matrix-vector product functions that are consistent with ground truth
    jvp_func, vjp_func = _create_consistent_matrix_vector_functions(
        func, inputs, input_shape, output_shape, input_dim, output_dim, debug
    )

    # Randomized SVD Algorithm (following Halko et al. exactly)

    # Step 1: Generate random test matrix
    Omega = torch.randn(input_dim, l, device=device, dtype=dtype)
    if stabilize:
        Omega, _ = _safe_qr_decomposition(Omega, debug)  # Use safe QR

    # Step 2: Form Y = A * Omega (where A is our Jacobian)
    Y = torch.zeros(output_dim, l, device=device, dtype=dtype)
    for i in range(l):
        Y[:, i] = jvp_func(Omega[:, i])

    # Step 3: Power iterations for improved accuracy
    for iteration in range(num_iter):
        # Orthogonalize Y
        if stabilize:
            Y, _ = _safe_qr_decomposition(Y, debug)  # Use safe QR

        # Z = A^T * Y
        Z = torch.zeros(input_dim, l, device=device, dtype=dtype)
        for i in range(l):
            Z[:, i] = vjp_func(Y[:, i])

        # Orthogonalize Z
        if stabilize:
            Z, _ = _safe_qr_decomposition(Z, debug)  # Use safe QR

        # Y = A * Z
        Y_new = torch.zeros(output_dim, l, device=device, dtype=dtype)
        for i in range(l):
            Y_new[:, i] = jvp_func(Z[:, i])
        Y = Y_new

    # Step 4: QR decomposition of Y
    Q, R = _safe_qr_decomposition(Y, debug)  # Use safe QR
    Q = Q[:, :k]  # Keep only first k columns

    # Step 5: Form the small matrix B = Q^T * A * Omega_k
    # This is the crucial step - we need to use the RIGHT Omega vectors
    # For best results, we should use an orthogonal set

    # Create a fresh set of orthogonal vectors for the final projection
    if l > k:
        # Use the first k columns of the original Omega, but ensure they're orthogonal
        Omega_k = Omega[:, :k]
        Omega_k, _ = torch.linalg.qr(Omega_k)
    else:
        Omega_k = Omega

    B = torch.zeros(k, k, device=device, dtype=dtype)
    for i in range(k):
        # Apply Jacobian to the i-th column of Omega_k
        y_i = jvp_func(Omega_k[:, i])
        # Project onto the subspace spanned by Q
        B[:, i] = Q.T @ y_i

    # Step 6: SVD of the small matrix B
    U_tilde, S, Vt_tilde = _safe_svd(B, debug)  # Use safe SVD

    # Step 7: Recover the singular vectors
    U = Q @ U_tilde  # Left singular vectors

    # Right singular vectors: V = Omega_k * V_tilde^T
    V = Omega_k @ Vt_tilde.T

    # Normalize right singular vectors
    for i in range(k):
        norm = torch.norm(V[:, i])
        if norm > 1e-10:
            V[:, i] = V[:, i] / norm

    if debug:
        print(f"Final shapes - U: {U.shape}, S: {S.shape}, V: {V.shape}")
        print(f"Singular values: {S}")

    return U, S, V


def _setup_dimensions(inputs, func, debug=False):
    """Setup device, dtype, and dimension information with automatic dtype handling."""

    if isinstance(inputs, torch.Tensor):
        inputs = inputs.requires_grad_(True)
        input_shape = inputs.shape
        input_dim = inputs.numel()
        device = inputs.device
        dtype = inputs.dtype
    else:
        inputs = [x.requires_grad_(True) for x in inputs]
        input_shape = [x.shape for x in inputs]
        input_dim = sum(x.numel() for x in inputs)
        device = inputs[0].device
        dtype = inputs[0].dtype

    # Get output dimensions using original inputs (don't convert yet)
    outputs = func(inputs)
    if isinstance(outputs, torch.Tensor):
        output_shape = outputs.shape
        output_dim = outputs.numel()
    else:
        output_shape = [out.shape for out in outputs]
        output_dim = sum(out.numel() for out in outputs)

    if debug:
        print(f"Input shape: {input_shape}, dim: {input_dim}")
        print(f"Output shape: {output_shape}, dim: {output_dim}")
        print(f"Using dtype: {dtype}")

    return device, dtype, input_dim, output_dim, input_shape, output_shape


def _create_consistent_matrix_vector_functions(func, inputs, input_shape, output_shape,
                                             input_dim, output_dim, debug=False):
    """
    Create JVP and VJP functions that are consistent with torch.autograd.functional.jacobian.
    Uses original dtype for model compatibility, converts only for linear algebra operations.
    """

    # Keep original dtype for model compatibility
    original_dtype = inputs.dtype if isinstance(inputs, torch.Tensor) else inputs[0].dtype
    needs_conversion = original_dtype in [torch.bfloat16, torch.float16]

    if needs_conversion and debug:
        print(f"Will convert to float32 only for linear algebra operations (model uses: {original_dtype})")

    # Create flattened function that matches jacobian() exactly
    def flattened_func(flat_inputs):
        if isinstance(inputs, torch.Tensor):
            shaped_inputs = flat_inputs.reshape(input_shape)
        else:
            shaped_inputs = []
            offset = 0
            for i, shape in enumerate(input_shape):
                size = torch.prod(torch.tensor(shape)).item()
                shaped_inputs.append(flat_inputs[offset:offset+size].reshape(shape))
                offset += size
            shaped_inputs = tuple(shaped_inputs)

        outputs = func(shaped_inputs)

        if isinstance(outputs, torch.Tensor):
            return outputs.flatten()
        else:
            return torch.cat([out.flatten() for out in outputs])

    # Create flattened inputs in original dtype
    if isinstance(inputs, torch.Tensor):
        flat_inputs = inputs.flatten().requires_grad_(True)
    else:
        flat_inputs = torch.cat([inp.flatten() for inp in inputs]).requires_grad_(True)

    def jacobian_vector_product(v):
        """J @ v using JVP - converts v to original dtype if needed"""
        if needs_conversion:
            v_model = v.to(original_dtype)
        else:
            v_model = v

        _, result = jvp(flattened_func, (flat_inputs,), (v_model,))

        # Convert result to float32 for linear algebra if needed
        if needs_conversion:
            result = result.float()
        return result

    def jacobian_transpose_vector_product(u):
        """J^T @ u using VJP - converts u to original dtype if needed"""
        if needs_conversion:
            u_model = u.to(original_dtype)
        else:
            u_model = u

        _, vjp_fn = vjp(flattened_func, flat_inputs)
        result = vjp_fn(u_model)

        # vjp_fn returns a tuple, extract first element
        if isinstance(result, tuple):
            result = result[0]

        # Convert result to float32 for linear algebra if needed
        if needs_conversion:
            result = result.float()
        return result

    if debug:
        print("Created consistent matrix-vector product functions with dtype handling")

    return jacobian_vector_product, jacobian_transpose_vector_product


def _safe_qr_decomposition(matrix, debug=False):
    """Perform QR decomposition with automatic dtype conversion for CUDA compatibility."""
    original_dtype = matrix.dtype
    needs_conversion = original_dtype in [torch.bfloat16, torch.float16]

    if needs_conversion:
        if debug:
            print(f"Converting to float32 for QR decomposition (original: {original_dtype})")
        matrix_float = matrix.float()
        Q, R = torch.linalg.qr(matrix_float)
        # Convert back to original dtype if needed (though we usually want float32 for subsequent ops)
        return Q, R
    else:
        return torch.linalg.qr(matrix)


def _safe_svd(matrix, debug=False):
    """Perform SVD with automatic dtype conversion for CUDA compatibility."""
    original_dtype = matrix.dtype
    needs_conversion = original_dtype in [torch.bfloat16, torch.float16]

    if needs_conversion:
        if debug:
            print(f"Converting to float32 for SVD (original: {original_dtype})")
        matrix_float = matrix.float()
        return torch.linalg.svd(matrix_float, full_matrices=False)
    else:
        return torch.linalg.svd(matrix, full_matrices=False)
    """
    Compute ground truth SVD using torch.autograd.functional.jacobian.
    WARNING: Only use for small problems!
    Handles dtype conversion automatically.
    """
    from torch.autograd.functional import jacobian

    print("Computing full Jacobian for ground truth...")

    # Handle dtype conversion
    original_dtype = inputs.dtype if isinstance(inputs, torch.Tensor) else inputs[0].dtype
    needs_conversion = original_dtype in [torch.bfloat16, torch.float16]

    if needs_conversion:
        print(f"Converting from {original_dtype} to float32 for ground truth computation")
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.float()
        else:
            inputs = [x.float() for x in inputs]

    # Create the same flattened function used in our randomized approach
    if isinstance(inputs, torch.Tensor):
        input_shape = inputs.shape
        input_dim = inputs.numel()
    else:
        input_shape = [x.shape for x in inputs]
        input_dim = sum(x.numel() for x in inputs)

    def flattened_func(flat_inputs):
        # Convert to original dtype for model execution
        if needs_conversion:
            flat_inputs = flat_inputs.to(original_dtype)

        if isinstance(inputs, torch.Tensor):
            shaped_inputs = flat_inputs.reshape(input_shape)
        else:
            shaped_inputs = []
            offset = 0
            for i, shape in enumerate(input_shape):
                size = torch.prod(torch.tensor(shape)).item()
                shaped_inputs.append(flat_inputs[offset:offset+size].reshape(shape))
                offset += size
            shaped_inputs = tuple(shaped_inputs)

        outputs = func(shaped_inputs)
        if isinstance(outputs, torch.Tensor):
            result = outputs.flatten()
        else:
            result = torch.cat([out.flatten() for out in outputs])

        # Convert back to float32 for linear algebra
        if needs_conversion:
            result = result.float()

        return result

    # Flatten inputs exactly as in randomized version
    if isinstance(inputs, torch.Tensor):
        flat_inputs = inputs.flatten().requires_grad_(True)
    else:
        flat_inputs = torch.cat([inp.flatten() for inp in inputs]).requires_grad_(True)

    # Ensure we're using float32 for linear algebra
    if needs_conversion:
        flat_inputs = flat_inputs.float().requires_grad_(True)

    # Compute Jacobian
    jac = jacobian(flattened_func, flat_inputs, vectorize=True)
    print(f"Full Jacobian shape: {jac.shape}")

    # Compute SVD
    U_full, S_full, Vt_full = torch.linalg.svd(jac, full_matrices=False)

    # Return only top k if specified
    if num_singular_vectors is not None:
        k = min(num_singular_vectors, len(S_full))
        return U_full[:, :k], S_full[:k], Vt_full[:k].T
    else:
        return U_full, S_full, Vt_full.T


def validate_randomized_svd(func, inputs, num_singular_vectors=3, test_ground_truth=True):
    """
    Comprehensive validation of the randomized SVD implementation.
    """
    print("="*60)
    print("VALIDATING RANDOMIZED SVD IMPLEMENTATION")
    print("="*60)

    # Get problem dimensions
    if isinstance(inputs, torch.Tensor):
        input_dim = inputs.numel()
    else:
        input_dim = sum(x.numel() for x in inputs)

    outputs = func(inputs)
    if isinstance(outputs, torch.Tensor):
        output_dim = outputs.numel()
    else:
        output_dim = sum(out.numel() for out in outputs)

    print(f"Problem size: {output_dim} × {input_dim}")

    # Test our improved implementation
    print("\n1. Computing with improved randomized SVD...")
    U_rand, S_rand, V_rand = randomized_svd_jacobian_improved(
        func, inputs,
        num_singular_vectors=num_singular_vectors,
        num_iter=4,
        oversampling=10,
        debug=False,
        stabilize=True
    )
    print(f"Randomized SVD singular values: {S_rand.detach().numpy()}")

    # Test against ground truth if problem is small enough
    if test_ground_truth and input_dim * output_dim < 50000:
        print("\n2. Computing ground truth...")
        try:
            U_true, S_true, V_true = compute_ground_truth_svd(func, inputs, num_singular_vectors)
            print(f"Ground truth singular values:   {S_true.detach().numpy()}")

            print("\n3. ACCURACY ANALYSIS:")
            print("-" * 40)

            # Singular value errors
            s_error = torch.abs(S_rand - S_true) / (S_true + 1e-10)
            print(f"Relative singular value errors: {s_error.detach().numpy()}")
            print(f"Max relative error: {s_error.max().item():.6f}")

            # Subspace alignment (how well singular vectors align)
            subspace_errors = []
            for i in range(min(num_singular_vectors, len(S_rand), len(S_true))):
                # Check alignment of both left and right singular vectors
                u_alignment = torch.abs(torch.dot(U_rand[:, i], U_true[:, i]))
                v_alignment = torch.abs(torch.dot(V_rand[:, i], V_true[:, i]))
                # Subspace error is 1 - alignment (0 = perfect, 1 = orthogonal)
                subspace_error = max(1 - u_alignment.item(), 1 - v_alignment.item())
                subspace_errors.append(subspace_error)

            print(f"Subspace alignment errors: {subspace_errors}")
            print(f"Max subspace error: {max(subspace_errors):.6f}")

            # Overall assessment
            if s_error.max().item() < 0.01 and max(subspace_errors) < 0.1:
                print("\n✅ EXCELLENT: High accuracy achieved!")
            elif s_error.max().item() < 0.1 and max(subspace_errors) < 0.3:
                print("\n✅ GOOD: Reasonable accuracy for randomized method")
            else:
                print("\n⚠️  NEEDS IMPROVEMENT: Consider more iterations or oversampling")

        except Exception as e:
            print(f"\n⚠️  Ground truth computation failed: {e}")
            print("This is expected for large problems - randomized method is the only option.")

    else:
        print("\n⚠️  Skipping ground truth (problem too large or disabled)")
        print("Randomized method completed successfully.")

    print("\n4. PERFORMANCE INSIGHTS:")
    print("-" * 40)
    print(f"• Memory usage: O({max(input_dim, output_dim)} × {num_singular_vectors + 10})")
    print(f"• Computational cost: ~{4 * (num_singular_vectors + 10)} matrix-vector products")
    print(f"• Full SVD would require: O({input_dim * output_dim}) memory")

    return U_rand, S_rand, V_rand


def randomized_svd_jacobian_v2(func, inputs, num_singular_vectors=5, num_iter=4,
                              oversampling=10, debug=False, use_full_for_small=True):
    """
    Alternative implementation that uses full SVD for small problems
    and a more robust randomized approach for larger ones.
    """

    # Setup and dimension calculation
    device, dtype, input_dim, output_dim, input_shape, output_shape = _setup_dimensions(inputs, func, debug)

    k = num_singular_vectors

    # For small problems, just compute the full Jacobian
    # if use_full_for_small and input_dim * output_dim < 1000:
    #     if debug:
    #         print("Using full SVD for small problem")
    #     return compute_ground_truth_svd(func, inputs, num_singular_vectors)

    # For larger problems, use randomized SVD with better parameters
    if debug:
        print("Using randomized SVD for large problem")

    # More aggressive settings for better accuracy
    oversampling = max(oversampling, k * 2)  # At least 2x oversampling
    l = min(k + oversampling, min(input_dim, output_dim))

    # Create matrix-vector product functions
    jvp_func, vjp_func = _create_consistent_matrix_vector_functions(
        func, inputs, input_shape, output_shape, input_dim, output_dim, debug
    )

    # Enhanced randomized SVD with subspace iteration

    # Step 1: Generate random test matrix with better conditioning
    Omega = torch.randn(input_dim, l, device=device, dtype=dtype)
    Omega, _ = _safe_qr_decomposition(Omega, debug)  # Use safe QR

    # Step 2: Form Y = A * Omega
    Y = torch.zeros(output_dim, l, device=device, dtype=dtype)
    for i in range(l):
        Y[:, i] = jvp_func(Omega[:, i])

    # Step 3: Enhanced power iterations with subspace iteration
    for iteration in range(num_iter):
        # Orthogonalize Y
        Y, _ = _safe_qr_decomposition(Y, debug)  # Use safe QR

        # Z = A^T * Y
        Z = torch.zeros(input_dim, l, device=device, dtype=dtype)
        for i in range(l):
            Z[:, i] = vjp_func(Y[:, i])

        # Orthogonalize Z
        Z, _ = _safe_qr_decomposition(Z, debug)  # Use safe QR

        # Y = A * Z (subspace iteration)
        Y = torch.zeros(output_dim, l, device=device, dtype=dtype)
        for i in range(l):
            Y[:, i] = jvp_func(Z[:, i])

    # Step 4: Final range approximation
    Q, _ = _safe_qr_decomposition(Y, debug)  # Use safe QR
    Q = Q[:, :k]

    # Step 5: Project Jacobian onto subspace - use the final Z vectors
    # This gives us B = Q^T * A * Z_k where Z_k spans the right subspace
    Z_k = Z[:, :k]

    B = torch.zeros(k, k, device=device, dtype=dtype)
    for i in range(k):
        y_i = jvp_func(Z_k[:, i])
        B[:, i] = Q.T @ y_i

    # Step 6: SVD of B
    U_tilde, S, Vt_tilde = _safe_svd(B, debug)  # Use safe SVD

    # Step 7: Recover singular vectors
    U = Q @ U_tilde
    V = Z_k @ Vt_tilde.T

    # Normalize V
    for i in range(k):
        norm = torch.norm(V[:, i])
        if norm > 1e-10:
            V[:, i] = V[:, i] / norm

    if debug:
        print(f"V2 Final shapes - U: {U.shape}, S: {S.shape}, V: {V.shape}")
        print(f"V2 Singular values: {S}")

    return U, S, V


def run_comprehensive_tests():
    """Run comprehensive tests on various problem types."""

    print("🧪 COMPREHENSIVE RANDOMIZED SVD TESTS")
    print("="*60)

    # Test 1: Simple linear function
    print("\nTEST 1: Linear transformation")
    print("-" * 30)

    def linear_func(x):
        W = torch.tensor([[2.0, -1.0, 0.5], [1.0, 1.0, -1.0]], dtype=torch.float32)
        return x @ W.T

    x1 = torch.randn(3, 3, dtype=torch.float32, requires_grad=True)

    print("Original implementation:")
    validate_randomized_svd(linear_func, x1, num_singular_vectors=2, test_ground_truth=True)

    print("\nV2 implementation:")
    U_v2, S_v2, V_v2 = randomized_svd_jacobian_v2(linear_func, x1, num_singular_vectors=2, debug=False)
    print(f"V2 Singular values: {S_v2.detach().numpy()}")

    # Test 2: Nonlinear function
    print("\n\nTEST 2: Nonlinear neural network")
    print("-" * 30)

    def nonlinear_func(x):
        W1 = torch.tensor([[1.0, -0.5], [0.5, 1.0], [-1.0, 0.5]], dtype=torch.float32)
        b1 = torch.tensor([0.1, -0.1, 0.2], dtype=torch.float32)
        h = torch.tanh(x @ W1.T + b1)

        W2 = torch.tensor([[1.0, 0.5, -1.0], [0.0, 1.0, 0.5]], dtype=torch.float32)
        b2 = torch.tensor([0.0, 0.1], dtype=torch.float32)
        return h @ W2.T + b2

    x2 = torch.randn(4, 2, dtype=torch.float32, requires_grad=True)

    print("Original implementation:")
    validate_randomized_svd(nonlinear_func, x2, num_singular_vectors=2, test_ground_truth=True)

    print("\nV2 implementation:")
    U_v2, S_v2, V_v2 = randomized_svd_jacobian_v2(nonlinear_func, x2, num_singular_vectors=2, debug=False)
    print(f"V2 Singular values: {S_v2.detach().numpy()}")

    # Test 3: Larger problem (no ground truth)
    print("\n\nTEST 3: Larger problem (randomized only)")
    print("-" * 30)

    def large_func(x):
        W1 = torch.randn(30, 50, dtype=torch.float32)
        b1 = torch.randn(30, dtype=torch.float32)
        h = F.relu(x @ W1.T + b1)

        W2 = torch.randn(10, 30, dtype=torch.float32)
        b2 = torch.randn(10, dtype=torch.float32)
        return torch.sigmoid(h @ W2.T + b2)

    x3 = torch.randn(8, 50, dtype=torch.float32, requires_grad=True)

    print("V2 implementation (randomized for large problem):")
    U_v2, S_v2, V_v2 = randomized_svd_jacobian_v2(large_func, x3, num_singular_vectors=5,
                                                   debug=False, use_full_for_small=False)
    print(f"V2 Singular values: {S_v2.detach().numpy()}")


def get_gpu_memory_info():
    """Get current GPU memory usage information."""
    if not torch.cuda.is_available():
        return {"available": False}

    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory = torch.cuda.memory_reserved(device)
    free_memory = total_memory - allocated_memory

    return {
        "available": True,
        "device": device,
        "total_gb": total_memory / (1024**3),
        "allocated_gb": allocated_memory / (1024**3),
        "cached_gb": cached_memory / (1024**3),
        "free_gb": free_memory / (1024**3),
        "utilization": allocated_memory / total_memory
    }


def jacobian_svd(func, inputs, num_singular_vectors=5, create_graph=False, strict=False,
                 vectorize=False, strategy='auto', disable_flash_attn=True, debug=False,
                 max_memory_gb=8, gpu_memory_fraction=0.8, per_token=False,
                 return_numpy=True, **svd_kwargs):
    """
    Compute top singular vectors and values of the Jacobian matrix without materializing the full Jacobian.

    This function mimics the interface of torch.autograd.functional.jacobian but returns
    singular value decomposition results instead of the full Jacobian matrix.

    Args:
        func: A Python function that takes Tensor inputs and returns
              a Tensor or tuple of Tensors
        inputs: Tensor or tuple of Tensors that are inputs to `func`
        num_singular_vectors: Number of top singular vectors to compute (default: 5)
        create_graph: If True, the Jacobian will be computed in a differentiable manner (not implemented)
        strict: If True, an error will be raised when we detect that there exists an input
               such that all the outputs are independent of it (not implemented)
        vectorize: This flag is experimental, use at your own risk (ignored)
        strategy: 'auto', 'full', or 'randomized'
                 - 'auto': automatically choose based on problem size
                 - 'full': always compute full Jacobian first (memory intensive)
                 - 'randomized': always use randomized method
        disable_flash_attn: Whether to disable Flash Attention for JVP/VJP compatibility (default: True)
        debug: Enable debugging output (default: False)
        max_memory_gb: Maximum memory to use for full Jacobian (default: 8GB)
        gpu_memory_fraction: Fraction of free GPU memory to use (default: 0.8)
        per_token: If True, return separate SVD for each token position (default: False)
        return_numpy: If True, return numpy arrays instead of tensors (saves GPU memory)
        **svd_kwargs: Additional arguments passed to the SVD computation
                     (num_iter, oversampling, debug, etc.)

    Returns:
        JacobianSVD: A named tuple containing:
            - U: Left singular vectors (output space) [output_dim x num_singular_vectors]
                 OR [num_tokens, output_dim, num_singular_vectors] if per_token=True
            - S: Singular values [num_singular_vectors]
                 OR [num_tokens, num_singular_vectors] if per_token=True
            - V: Right singular vectors (input space) [input_dim x num_singular_vectors]
                 OR [num_tokens, token_dim, num_singular_vectors] if per_token=True
            - input_dim: Flattened input dimension
            - output_dim: Flattened output dimension
            - num_tokens: Number of tokens (only if per_token=True)

    Example:
        >>> # Standard usage
        >>> svd_result = jacobian_svd(model, x, num_singular_vectors=3)

        >>> # Per-token analysis for LLMs
        >>> svd_result = jacobian_svd(model, embeddings, num_singular_vectors=5, per_token=True)
        >>> print(f"Token 0 singular values: {svd_result.S[0]}")
        >>> print(f"Token 1 singular values: {svd_result.S[1]}")
    """

    from collections import namedtuple
    import gc

    # Define return type (updated for per-token support)
    if per_token:
        JacobianSVD = namedtuple('JacobianSVD', ['U', 'S', 'V', 'input_dim', 'output_dim', 'num_tokens'])
    else:
        JacobianSVD = namedtuple('JacobianSVD', ['U', 'S', 'V', 'input_dim', 'output_dim'])

    # Handle create_graph and strict warnings
    if create_graph:
        print("Warning: create_graph=True is not yet implemented for jacobian_svd")
    if strict:
        print("Warning: strict=True is not yet implemented for jacobian_svd")

    # Safety check for reasonable problem size
    try:
        device, dtype, input_dim, output_dim, input_shape, output_shape = _setup_dimensions(inputs, func, debug=debug)
        if debug:
            print(f"Problem dimensions: {output_dim} × {input_dim}")
            print(f"Input shape: {input_shape}")
            print(f"Output shape: {output_shape}")
            print(f"Input dtype: {dtype}")
    except Exception as e:
        print(f"Error during dimension setup: {e}")
        raise

    # Per-token analysis for LLMs
    if per_token:
        if debug:
            print("Performing per-token Jacobian analysis...")

        # Detect token structure from input shape
        if isinstance(inputs, torch.Tensor) and len(input_shape) == 3:
            # Shape: [batch, num_tokens, token_dim]
            batch_size, num_tokens, token_dim = input_shape
            if batch_size != 1:
                raise ValueError(f"Per-token analysis requires batch_size=1, got {batch_size}")

            if debug:
                print(f"Detected {num_tokens} tokens, each with {token_dim} dimensions")

            # Compute SVD for each token position
            U_list, S_list, V_list = [], [], []

            for token_idx in range(num_tokens):
                if debug:
                    print(f"Computing SVD for token {token_idx}/{num_tokens}")
                    if torch.cuda.is_available():
                        gpu_info = get_gpu_memory_info()
                        print(f"  Before token {token_idx}: GPU free = {gpu_info['free_gb']:.1f}GB")

                # Create function that only varies the specific token
                def token_specific_func(token_embedding):
                    # token_embedding shape: [token_dim]
                    full_inputs = inputs.clone()
                    full_inputs[0, token_idx, :] = token_embedding
                    return func(full_inputs)

                # Extract the specific token embedding
                token_input = inputs[0, token_idx, :].requires_grad_(True)

                # Compute SVD for this token
                token_result = _compute_single_svd(
                    token_specific_func, token_input, num_singular_vectors,
                    strategy, debug, max_memory_gb, gpu_memory_fraction,
                    token_dim, output_dim, disable_flash_attn, **svd_kwargs
                )

                # Convert to numpy immediately to save GPU memory
                U_numpy = token_result.U.detach().cpu().numpy()
                S_numpy = token_result.S.detach().cpu().numpy()
                V_numpy = token_result.V.detach().cpu().numpy()

                U_list.append(U_numpy)
                S_list.append(S_numpy)
                V_list.append(V_numpy)

                # Aggressive cleanup after each token
                del token_result, U_numpy, S_numpy, V_nummy
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Ensure all operations complete

                if debug:
                    gpu_info = get_gpu_memory_info()
                    print(f"  After token {token_idx}: GPU free = {gpu_info['free_gb']:.1f}GB")

            # Convert numpy arrays back to tensors only at the end
            device = inputs.device
            original_dtype = inputs.dtype

            # Stack numpy arrays first (more memory efficient)
            U_numpy_stacked = np.stack(U_list, axis=0)  # [num_tokens, output_dim, k]
            S_numpy_stacked = np.stack(S_list, axis=0)  # [num_tokens, k]
            V_numpy_stacked = np.stack(V_list, axis=0)  # [num_tokens, token_dim, k]

            # Clean up intermediate lists
            del U_list, S_list, V_list
            gc.collect()

            # Convert back to tensors in original dtype to minimize memory
            # U_stacked = torch.from_numpy(U_numpy_stacked).to(device=device, dtype=original_dtype)
            # S_stacked = torch.from_numpy(S_numpy_stacked).to(device=device, dtype=original_dtype)
            # V_stacked = torch.from_numpy(V_numpy_stacked).to(device=device, dtype=original_dtype)
            U_stacked = torch.from_numpy(U_numpy_stacked).to(ddtype=original_dtype)
            S_stacked = torch.from_numpy(S_numpy_stacked).to(dtype=original_dtype)
            V_stacked = torch.from_numpy(V_numpy_stacked).to(dtype=original_dtype)

            # Clean up numpy arrays
            del U_numpy_stacked, S_numpy_stacked, V_numpy_stacked
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if debug:
                print(f"Per-token results - U: {U_stacked.shape}, S: {S_stacked.shape}, V: {V_stacked.shape}")

            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return JacobianSVD(U=U_stacked, S=S_stacked, V=V_stacked,
                             input_dim=input_dim, output_dim=output_dim, num_tokens=num_tokens)

        else:
            raise ValueError(f"Per-token analysis requires 3D input tensor [batch, tokens, dim], got shape {input_shape}")

    # Standard analysis (original behavior)
    else:
        return _compute_single_svd(
            func, inputs, num_singular_vectors, strategy, debug,
            max_memory_gb, gpu_memory_fraction, input_dim, output_dim,
            disable_flash_attn, **svd_kwargs
        )


def _compute_single_svd(func, inputs, num_singular_vectors, strategy, debug,
                       max_memory_gb, gpu_memory_fraction, input_dim, output_dim,
                       disable_flash_attn, **svd_kwargs):
    """Helper function to compute SVD for a single input (used by both modes)."""
    from collections import namedtuple
    import gc

    JacobianSVD = namedtuple('JacobianSVD', ['U', 'S', 'V', 'input_dim', 'output_dim'])

    # Get device info
    device = inputs.device if isinstance(inputs, torch.Tensor) else inputs[0].device

    # Memory safety check
    jacobian_memory_gb = (input_dim * output_dim * 4) / (1024**3)  # 4 bytes per float32

    # GPU memory check if using CUDA
    if torch.cuda.is_available() and device.type == 'cuda':
        gpu_info = get_gpu_memory_info()
        if debug:
            print(f"GPU Memory - Total: {gpu_info['total_gb']:.1f}GB, "
                  f"Free: {gpu_info['free_gb']:.1f}GB, "
                  f"Utilization: {gpu_info['utilization']:.1%}")
            print(f"Full Jacobian would require ~{jacobian_memory_gb:.2f}GB")

        # Use GPU memory limit if more restrictive
        gpu_available = gpu_info['free_gb'] * gpu_memory_fraction
        effective_memory_limit = min(max_memory_gb, gpu_available)

        if debug:
            print(f"Effective memory limit: {effective_memory_limit:.1f}GB "
                  f"(GPU: {gpu_available:.1f}GB, RAM: {max_memory_gb}GB)")
    else:
        effective_memory_limit = max_memory_gb
        if debug:
            print(f"Using CPU, memory limit: {effective_memory_limit:.1f}GB")
            print(f"Full Jacobian would require ~{jacobian_memory_gb:.2f}GB")

    # Wrap computation with Flash Attention handling
    def compute_with_attention_handling():
        # Get default SVD parameters
        default_svd_params = {
            'num_iter': 4,
            'oversampling': min(10, min(input_dim, output_dim) // 2),  # Safety limit
            'debug': debug
        }
        default_svd_params.update(svd_kwargs)

        if debug:
            print(f"SVD parameters: {default_svd_params}")

        # Choose strategy with safety checks
        if strategy == 'auto':
            if jacobian_memory_gb > effective_memory_limit:
                strategy_chosen = 'randomized'
                if debug:
                    print(f"Auto-choosing randomized (memory: {jacobian_memory_gb:.2f}GB > {effective_memory_limit:.1f}GB)")
            else:
                strategy_chosen = 'full'
                if debug:
                    print(f"Auto-choosing full SVD (memory: {jacobian_memory_gb:.2f}GB <= {effective_memory_limit:.1f}GB)")
        else:
            strategy_chosen = strategy
            if debug:
                print(f"Using user-specified strategy: {strategy_chosen}")

        # Safety check for randomized method
        if strategy_chosen == 'randomized' and num_singular_vectors >= min(input_dim, output_dim):
            print(f"Warning: num_singular_vectors ({num_singular_vectors}) >= min(input_dim, output_dim) ({min(input_dim, output_dim)})")
            num_singular_vectors_safe = min(num_singular_vectors, min(input_dim, output_dim) - 1)
            print(f"Reducing to {num_singular_vectors_safe}")
        else:
            num_singular_vectors_safe = num_singular_vectors

        try:
            # Compute SVD based on strategy
            if strategy_chosen == 'full':
                if debug:
                    print("Computing full SVD...")
                U, S, V = compute_ground_truth_svd(func, inputs, num_singular_vectors_safe)

            elif strategy_chosen == 'randomized':
                if debug:
                    print("Computing randomized SVD...")
                U, S, V = randomized_svd_jacobian_v2(func, inputs, num_singular_vectors_safe,
                                                   use_full_for_small=False, **default_svd_params)
            else:
                raise ValueError(f"Unknown strategy: {strategy_chosen}. Must be 'auto', 'full', or 'randomized'")

            if debug:
                print(f"SVD computation successful!")
                print(f"Returned shapes: U={U.shape}, S={S.shape}, V={V.shape}")

            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return JacobianSVD(U=U, S=S, V=V, input_dim=input_dim, output_dim=output_dim)

        except Exception as e:
            if debug:
                print(f"Error during SVD computation: {e}")
            if strategy_chosen == 'full':
                print(f"Full SVD failed ({e}), falling back to randomized method")
                try:
                    U, S, V = randomized_svd_jacobian_v2(func, inputs, num_singular_vectors_safe,
                                                       use_full_for_small=False, **default_svd_params)
                    return JacobianSVD(U=U, S=S, V=V, input_dim=input_dim, output_dim=output_dim)
                except Exception as e2:
                    print(f"Randomized SVD also failed: {e2}")
                    raise
            else:
                raise

    # Execute with or without Flash Attention disabled
    try:
        if disable_flash_attn:
            if debug:
                print("Disabling Flash Attention...")
            with disable_flash_attention():
                return compute_with_attention_handling()
        else:
            if debug:
                print("Using existing attention settings...")
            return compute_with_attention_handling()
    except Exception as e:
        print(f"Fatal error in jacobian_svd: {e}")
        # Clean up memory before re-raising
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        raise


def demo_jacobian_svd_interface():
    """Demonstrate the jacobian_svd interface with examples similar to torch.autograd.functional.jacobian"""

    print("🚀 JACOBIAN SVD INTERFACE DEMO")
    print("="*50)

    # Example 1: Simple function like you'd use with torch.autograd.functional.jacobian
    print("\nExample 1: Linear transformation")
    print("-" * 30)

    def linear_model(x):
        # Simple element-wise transformation that's easy to understand
        return torch.stack([
            x.sum(dim=1),           # Sum across features
            (x**2).mean(dim=1),     # Mean of squares
            x[:, 0] - x[:, 1]       # Simple difference
        ], dim=1)

    x = torch.randn(4, 3, dtype=torch.float32, requires_grad=True)  # [batch=4, features=3]

    # Let's skip the full Jacobian comparison for simplicity and just show the interface
    print("Using jacobian_svd:")
    svd_result = jacobian_svd(linear_model, x, num_singular_vectors=3, debug=False)
    print(f"  Input dim: {svd_result.input_dim}, Output dim: {svd_result.output_dim}")
    print(f"  Top 3 singular values: {svd_result.S}")
    print(f"  Left vectors shape: {svd_result.U.shape}")
    print(f"  Right vectors shape: {svd_result.V.shape}")

    # Example 2: Neural network
    print("\n\nExample 2: Neural network")
    print("-" * 30)

    def neural_net(x):
        W1 = torch.randn(20, 10, dtype=torch.float32)
        b1 = torch.randn(20, dtype=torch.float32)
        h = F.relu(x @ W1.T + b1)

        W2 = torch.randn(5, 20, dtype=torch.float32)
        b2 = torch.randn(5, dtype=torch.float32)
        return h @ W2.T + b2

    x_large = torch.randn(8, 10, dtype=torch.float32, requires_grad=True)

    # With jacobian_svd (memory efficient)
    svd_result = jacobian_svd(neural_net, x_large, num_singular_vectors=5, strategy='randomized')
    print(f"  Problem size: {svd_result.output_dim} × {svd_result.input_dim}")
    print(f"  Top 5 singular values: {svd_result.S}")
    print(f"  Full Jacobian would need: {svd_result.input_dim * svd_result.output_dim} elements")
    print(f"  SVD result uses only: {svd_result.U.numel() + svd_result.V.numel() + svd_result.S.numel()} elements")
    memory_savings = (svd_result.input_dim * svd_result.output_dim) / (svd_result.U.numel() + svd_result.V.numel() + svd_result.S.numel())
    print(f"  Memory savings: {memory_savings:.1f}x less memory!")

    # Example 3: Demonstrate different strategies
    print("\n\nExample 3: Strategy comparison")
    print("-" * 30)

    def small_function(x):
        return torch.cat([
            x.sum(dim=1, keepdim=True),
            x.mean(dim=1, keepdim=True),
            x.std(dim=1, keepdim=True)
        ], dim=1)

    x_small = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)

    # Auto strategy (will choose full for small problem)
    auto_result = jacobian_svd(small_function, x_small, num_singular_vectors=2, strategy='auto')
    print(f"  Auto strategy singular values: {auto_result.S}")

    # Forced randomized strategy
    rand_result = jacobian_svd(small_function, x_small, num_singular_vectors=2, strategy='randomized')
    print(f"  Randomized strategy singular values: {rand_result.S}")

    # They should be very close for this small problem
    error = torch.abs(auto_result.S - rand_result.S) / (auto_result.S + 1e-10)
    print(f"  Relative error between strategies: {error}")

    # Example 4: Show how to replace torch.autograd.functional.jacobian usage
    print("\n\nExample 4: Typical usage patterns")
    print("-" * 30)

    def some_model(params):
        # Simulate a model that takes parameters and outputs predictions
        x = torch.randn(5, 3)  # Some fixed input data
        return F.linear(x, params.reshape(2, 3))  # params as weights

    params = torch.randn(6, requires_grad=True)  # 6 parameters -> 2x3 weight matrix

    print("  Instead of:")
    print("    jac = torch.autograd.functional.jacobian(some_model, params)")
    print("    U, S, V = torch.linalg.svd(jac)")
    print("  ")
    print("  Use:")
    print("    svd_result = jacobian_svd(some_model, params, num_singular_vectors=3)")
    print("    U, S, V = svd_result.U, svd_result.S, svd_result.V")

    # Actually compute it
    svd_result = jacobian_svd(some_model, params, num_singular_vectors=3)
    print(f"  ")
    print(f"  Result: Top 3 singular values = {svd_result.S}")
    print(f"  Jacobian shape would be: {svd_result.output_dim} × {svd_result.input_dim}")

    print("\n✅ Demo complete! You can now use jacobian_svd() as a drop-in replacement")
    print("   for scenarios where you only need the top singular vectors of the Jacobian.")


# if __name__ == "__main__":
#     run_comprehensive_tests()
#     demo_jacobian_svd_interface()
#     print("\n" + "="*60)

####@title Vectorized SVD with memory cleanup 2

import torch
import torch.nn.functional as F
from torch.func import jvp, vjp, vmap
import numpy as np
import math
from contextlib import contextmanager
import time

def randomized_svd_jacobian_vectorized(func, inputs, num_singular_vectors=5, num_iter=4,
                                     oversampling=10, debug=False, stabilize=True):
    """
    Vectorized randomized SVD for Jacobian matrices using vmap for JVP/VJP operations.
    Includes strategic memory cleanup to reduce GPU memory usage.
    Handles bfloat16 and float16 inputs by converting to float32 for linear algebra.

    Args:
        func: Function whose Jacobian we want to analyze
        inputs: Input tensor(s) to the function
        num_singular_vectors: Number of top singular vectors to compute
        num_iter: Number of power iterations for accuracy
        oversampling: Extra random vectors for stability
        debug: Print debugging information
        stabilize: Use numerical stabilization techniques

    Returns:
        U: Left singular vectors (output space)
        S: Singular values
        V: Right singular vectors (input space)
    """
    import gc

    # Use disable_flash_attention context manager for JVP/VJP compatibility
    with disable_flash_attention():
        # Setup and dimension calculation
        device, dtype, input_dim, output_dim, input_shape, output_shape = _setup_dimensions(inputs, func, debug)

        # Check if we need to convert from low precision
        needs_conversion = dtype in [torch.bfloat16, torch.float16]
        # Use float32 for linear algebra operations if original dtype is bfloat16 or float16
        computation_dtype = torch.float32 if needs_conversion else dtype

        if debug and needs_conversion:
            print(f"Converting from {dtype} to {computation_dtype} for linear algebra operations")

        # Memory cleanup after dimension setup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        k = num_singular_vectors

        # Adaptive parameters for small problems
        if min(input_dim, output_dim) < 20:
            # For very small problems, use more aggressive settings
            oversampling = max(oversampling, min(input_dim, output_dim) - k)
            num_iter = max(num_iter, 6)
            if debug:
                print(f"Small problem detected: using oversampling={oversampling}, num_iter={num_iter}")

        l = min(k + oversampling, min(input_dim, output_dim))

        if debug:
            print(f"Input dim: {input_dim}, Output dim: {output_dim}, k={k}, l={l}")

        # Create vectorized matrix-vector product functions
        jvp_vmap, vjp_vmap = _create_vectorized_matrix_vector_functions(
            func, inputs, input_shape, output_shape, input_dim, output_dim, debug
        )

        # Randomized SVD Algorithm with vmap optimization

        # Step 1: Generate random test matrix - use computation_dtype for numerical stability
        Omega = torch.randn(input_dim, l, device=device, dtype=computation_dtype)

        if stabilize:
            # QR decomposition requires at least float32
            Omega, _ = torch.linalg.qr(Omega)
        # Force CUDA synchronization
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.synchronize()
        # Memory cleanup after initialization
        del _ 
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 2: Form Y = A * Omega using vectorized JVP
        # Convert Omega to original dtype for the model operation
        Y = jvp_vmap(Omega.to(dtype))

        # Convert result to computation_dtype for numerical stability
        if needs_conversion:
            Y = Y.to(computation_dtype)

        # Free Omega if it's not needed anymore
        del Omega
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 3: Power iterations for improved accuracy
        for iteration in range(num_iter):
            if debug and iteration > 0 and iteration % 2 == 0:
                print(f"Completed {iteration}/{num_iter} power iterations")

            # Orthogonalize Y
            if stabilize:
                Y, R = torch.linalg.qr(Y)
                del R  # Free R matrix
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Z = A^T * Y using vectorized VJP
            # Convert Y to original dtype for the model operation
            Z = vjp_vmap(Y.to(dtype))

            # Convert result to computation_dtype for numerical stability
            if needs_conversion:
                Z = Z.to(computation_dtype)

            # Free Y before creating new Y
            del Y
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Orthogonalize Z
            if stabilize:
                Z, R = torch.linalg.qr(Z)
                del R  # Free R matrix
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Y = A * Z using vectorized JVP
            # Convert Z to original dtype for the model operation
            Y = jvp_vmap(Z.to(dtype))

            # Convert result to computation_dtype for numerical stability
            if needs_conversion:
                Y = Y.to(computation_dtype)

            # Free Z after creating Y
            del Z
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Step 4: QR decomposition of Y
        Q, R = torch.linalg.qr(Y)

        # Free Y and R
        del Y, R
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Keep only first k columns
        Q = Q[:, :k]

        # Step 5: Form the small matrix B = Q^T * A * Omega_k
        # Create a fresh set of orthogonal vectors for the final projection
        Omega_k = torch.randn(input_dim, k, device=device, dtype=computation_dtype)
        Omega_k, _ = torch.linalg.qr(Omega_k)

        # Apply JVP to all columns of Omega_k at once
        # Convert Omega_k to original dtype for the model operation
        Y_omega = jvp_vmap(Omega_k.to(dtype))

        # Convert result to computation_dtype for numerical stability
        if needs_conversion:
            Y_omega = Y_omega.to(computation_dtype)

        # Free Omega_k after use
        del Omega_k
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Project onto the subspace spanned by Q
        B = Q.T @ Y_omega  # Shape: [k, k]

        # Free Y_omega after use
        del Y_omega
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 6: SVD of the small matrix B
        U_tilde, S, Vt_tilde = torch.linalg.svd(B, full_matrices=False)

        # Free B after use
        del B
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 7: Recover the singular vectors
        U = Q @ U_tilde  # Left singular vectors

        # Free Q and U_tilde after use
        del Q, U_tilde
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 5 (again): For better right singular vectors, we need Omega_k again
        Omega_k = torch.randn(input_dim, k, device=device, dtype=computation_dtype)
        Omega_k, _ = torch.linalg.qr(Omega_k)

        # Right singular vectors: V = Omega_k * V_tilde^T
        V = Omega_k @ Vt_tilde.T

        # Free Omega_k and Vt_tilde after use
        del Omega_k, Vt_tilde
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Normalize right singular vectors
        V_norms = torch.norm(V, dim=0, keepdim=True)
        mask = V_norms > 1e-10
        V[:, mask.squeeze()] = V[:, mask.squeeze()] / V_norms[:, mask.squeeze()]

        # Free V_norms and mask
        del V_norms, mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Convert back to original dtype if requested
        if needs_conversion:
            U = U.to(dtype)
            S = S.to(dtype)
            V = V.to(dtype)

        if debug:
            print(f"Final shapes - U: {U.shape}, S: {S.shape}, V: {V.shape}")
            print(f"Singular values: {S}")

        return U, S, V
        del Omega_k, Vt_tilde
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Normalize right singular vectors
        V_norms = torch.norm(V, dim=0, keepdim=True)
        mask = V_norms > 1e-10
        V[:, mask.squeeze()] = V[:, mask.squeeze()] / V_norms[:, mask.squeeze()]

        # Free V_norms and mask
        del V_norms, mask

        U_clean, S_clean, V_clean = U.clone().detach(), S.clone().detach(), V.clone().detach()
        del U, S, V

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if debug:
            print(f"Final shapes - U: {U_clean.shape}, S: {S_clean.shape}, V: {V_clean.shape}")
            print(f"Singular values: {S_clean}")

        return U_clean, S_clean, V_clean


def randomized_svd_jacobian_per_token(func, inputs, num_singular_vectors=5, num_iter=4,
                                    oversampling=10, debug=False):
    """
    Compute SVD of Jacobian for each token position in a batched sequence.
    Uses vectorized operations for efficiency.
    Includes strategic memory cleanup to reduce GPU memory usage.

    Args:
        func: Function whose Jacobian we want to analyze (expected to take [batch, seq_len, emb_dim])
        inputs: Input tensor with shape [batch, seq_len, emb_dim]
        num_singular_vectors: Number of top singular vectors to compute per token
        num_iter: Number of power iterations for accuracy
        oversampling: Extra random vectors for stability
        debug: Print debugging information

    Returns:
        U_per_token: Left singular vectors for each token [seq_len, output_dim, k]
        S_per_token: Singular values for each token [seq_len, k]
        V_per_token: Right singular vectors for each token [seq_len, emb_dim, k]
    """
    import gc

    # Use disable_flash_attention context manager for JVP/VJP compatibility
    with disable_flash_attention():
        assert isinstance(inputs, torch.Tensor) and len(inputs.shape) == 3, \
            "Per-token analysis requires inputs of shape [batch, seq_len, emb_dim]"

        batch_size, seq_len, emb_dim = inputs.shape
        assert batch_size == 1, "Per-token analysis currently supports batch_size=1 only"

        if debug:
            print(f"Computing per-token Jacobian SVD for sequence length {seq_len}")
            print(f"Each token has embedding dimension {emb_dim}")

        # Initialize storage for results
        device = inputs.device
        dtype = inputs.dtype

        # First, get output dimensions by running the function once
        outputs = func(inputs)
        if isinstance(outputs, torch.Tensor):
            output_shape = outputs.shape
            output_dim = outputs.numel()
        else:
            raise ValueError("Per-token analysis expects tensor output from func")

        if debug:
            print(f"Output shape: {output_shape}, output_dim: {output_dim}")

        # Prepare containers for results
        U_per_token = []
        S_per_token = []
        V_per_token = []

        # Memory cleanup before token loop
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Define function that will be used for each token
        def token_func(token_idx, token_emb):
            # Create a copy of the inputs with the specified token replaced
            modified_inputs = inputs.clone()
            modified_inputs[0, token_idx, :] = token_emb
            return func(modified_inputs)

        # For each token position, compute the SVD of its Jacobian
        for token_idx in range(seq_len):
            if debug:
                print(f"Processing token {token_idx}/{seq_len}...")

            # Function that only varies this specific token's embedding
            def token_specific_func(token_emb):
                return token_func(token_idx, token_emb)

            # Extract the token embedding
            token_emb = inputs[0, token_idx, :].clone().requires_grad_(True)

            # Compute SVD for this token
            U, S, V = randomized_svd_jacobian_vectorized(
                token_specific_func, token_emb,
                num_singular_vectors=num_singular_vectors,
                num_iter=num_iter,
                oversampling=oversampling,
                debug=debug if token_idx == 0 else False,  # Only debug first token
                stabilize=True
            )

            # Store results and immediately move to CPU if needed to save GPU memory
            if device.type == 'cuda' and output_dim * emb_dim > 10000000:  # For very large matrices
                U_per_token.append(U.cpu())
                S_per_token.append(S.cpu())
                V_per_token.append(V.cpu())
                # Explicit cleanup
                del U, S, V
            else:
                U_per_token.append(U)
                S_per_token.append(S)
                V_per_token.append(V)

            # Aggressive cleanup after each token
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete

        # Stack results
        if device.type == 'cuda' and output_dim * emb_dim > 10000000:
            # First stack on CPU
            U_stacked = torch.stack(U_per_token, dim=0)
            S_stacked = torch.stack(S_per_token, dim=0)
            V_stacked = torch.stack(V_per_token, dim=0)
            # Then move back to GPU if needed
            # U_stacked = U_stacked.to(device)
            # S_stacked = S_stacked.to(device)
            # V_stacked = V_stacked.to(device)
        else:
            U_stacked = torch.stack(U_per_token, dim=0)  # [seq_len, output_dim, k]
            S_stacked = torch.stack(S_per_token, dim=0)  # [seq_len, k]
            V_stacked = torch.stack(V_per_token, dim=0)  # [seq_len, emb_dim, k]

        # Clean up intermediate lists
        del U_per_token, S_per_token, V_per_token
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if debug:
            print(f"Final shapes - U: {U_stacked.shape}, S: {S_stacked.shape}, V: {V_stacked.shape}")

        return U_stacked, S_stacked, V_stacked


def high_accuracy_svd(func, inputs, num_singular_vectors=5, num_iter=8, oversampling=None, debug=False):
    """
    Higher accuracy version of randomized SVD for capturing lower singular vectors more precisely.
    Uses more power iterations and adaptive oversampling for better accuracy.
    Handles bfloat16, float16, float32, and float64 inputs.

    Args:
        func: Function whose Jacobian we want to analyze
        inputs: Input tensor(s) to the function
        num_singular_vectors: Number of top singular vectors to compute
        num_iter: Number of power iterations for accuracy (default increased to 8)
        oversampling: Extra random vectors for stability (if None, uses 2*num_singular_vectors)
        debug: Print debugging information

    Returns:
        U: Left singular vectors (output space)
        S: Singular values
        V: Right singular vectors (input space)
    """
    import gc

    # Set adaptive oversampling if not specified
    if oversampling is None:
        oversampling = 2 * num_singular_vectors

    # Get dimensions and dtypes
    device, dtype, input_dim, output_dim, input_shape, output_shape = _setup_dimensions(inputs, func, debug)

    # Check if we need to convert from low precision
    needs_conversion = dtype in [torch.bfloat16, torch.float16]

    # Determine computation precision
    # For float32 inputs on reasonable-sized problems, use double precision
    # For bfloat16/float16, always convert to at least float32
    use_double = (dtype == torch.float32) and (input_dim * output_dim < 10000000)
    computation_dtype = torch.float64 if use_double else torch.float32

    if debug:
        print(f"Using {num_iter} power iterations and oversampling={oversampling}")
        if needs_conversion:
            print(f"Converting from {dtype} to {computation_dtype} for linear algebra operations")
        elif use_double:
            print("Using double precision for intermediate calculations")

    # Compute SVD with higher accuracy settings
    with disable_flash_attention():
        # Create vectorized matrix-vector product functions
        jvp_vmap, vjp_vmap = _create_vectorized_matrix_vector_functions(
            func, inputs, input_shape, output_shape, input_dim, output_dim, debug
        )

        # Memory cleanup after function creation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        k = num_singular_vectors
        l = min(k + oversampling, min(input_dim, output_dim))

        # Step 1: Generate random test matrix with orthogonalization
        # Always use at least float32 for numerical stability
        Omega = torch.randn(input_dim, l, device=device, dtype=computation_dtype)
        Omega, _ = torch.linalg.qr(Omega)  # Orthogonalize for better starting point

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 2: Form initial Y = A * Omega
        # Convert Omega to original dtype for jvp_vmap, which handles the original model
        Y = jvp_vmap(Omega.to(dtype))

        # Convert result back to computation precision
        Y = Y.to(computation_dtype)

        # Free Omega after use
        del Omega
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 3: Enhanced power iterations with subspace iteration
        for iteration in range(num_iter):
            if debug and iteration > 0 and iteration % 2 == 0:
                print(f"Completed {iteration}/{num_iter} power iterations")

            # Orthogonalize Y
            Y, _ = torch.linalg.qr(Y)

            # Z = A^T * Y
            # Convert Y to original dtype for vjp_vmap
            Z = vjp_vmap(Y.to(dtype))

            # Convert result back to computation precision
            Z = Z.to(computation_dtype)

            # Free Y before creating new Y
            del Y
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Orthogonalize Z
            Z, _ = torch.linalg.qr(Z)

            # Y = A * Z (subspace iteration)
            # Convert Z to original dtype for jvp_vmap
            Y = jvp_vmap(Z.to(dtype))

            # Convert result back to computation precision
            Y = Y.to(computation_dtype)

            # Free Z after creating Y
            del Z
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Step 4: Final QR of Y for more stable decomposition
        Q, _ = torch.linalg.qr(Y)

        # Free Y after use
        del Y
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Restrict to k columns
        Q = Q[:, :k]

        # Step 5: Create fresh set of orthogonal vectors for better right singular vectors
        Omega_k = torch.randn(input_dim, k, device=device, dtype=computation_dtype)
        Omega_k, _ = torch.linalg.qr(Omega_k)

        # Apply JVP to get Y_omega
        # Convert Omega_k to original dtype for jvp_vmap
        Y_omega = jvp_vmap(Omega_k.to(dtype))

        # Convert result back to computation precision
        Y_omega = Y_omega.to(computation_dtype)

        # Project onto Q subspace
        B = Q.T @ Y_omega

        # Free Y_omega after use
        del Y_omega
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 6: SVD of B
        U_tilde, S, Vt_tilde = torch.linalg.svd(B, full_matrices=False)

        # Free B after use
        del B
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 7: Recover singular vectors
        U = Q @ U_tilde
        V = Omega_k @ Vt_tilde.T

        # Free intermediate results
        del Q, U_tilde, Omega_k, Vt_tilde
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Convert back to original precision if requested
        if dtype != computation_dtype:
            # For inference only, user may want to keep in original dtype
            # But for randomized SVD, float32 is generally better for numerical stability
            U = U.to(dtype)
            S = S.to(dtype)
            V = V.to(dtype)

        # Normalize right singular vectors for numerical stability
        V_norms = torch.norm(V, dim=0, keepdim=True)
        mask = V_norms > 1e-10
        V[:, mask.squeeze()] = V[:, mask.squeeze()] / V_norms[:, mask.squeeze()]

        if debug:
            print(f"Final shapes - U: {U.shape}, S: {S.shape}, V: {V.shape}")
            print(f"Returned dtype: {U.dtype}")

        return U, S, V

def _create_vectorized_matrix_vector_functions(func, inputs, input_shape, output_shape, 
                                            input_dim, output_dim, debug=False):
    """
    Create vectorized JVP and VJP functions using vmap with aggressive memory management.
    
    This function creates JVP and VJP functions that operate on entire matrices at once,
    rather than processing each column individually, with careful memory cleanup.
    """
    import gc
    
    # Keep original dtype for model compatibility
    original_dtype = inputs.dtype if isinstance(inputs, torch.Tensor) else inputs[0].dtype
    needs_conversion = original_dtype in [torch.bfloat16, torch.float16]
    
    if needs_conversion and debug:
        print(f"Will convert to float32 only for linear algebra operations (model uses: {original_dtype})")
    
    # Create flattened function that matches jacobian() exactly
    def flattened_func(flat_inputs):
        try:
            if isinstance(inputs, torch.Tensor):
                shaped_inputs = flat_inputs.reshape(input_shape)
            else:
                shaped_inputs = []
                offset = 0
                for i, shape in enumerate(input_shape):
                    size = torch.prod(torch.tensor(shape)).item()
                    shaped_inputs.append(flat_inputs[offset:offset+size].reshape(shape))
                    offset += size
                shaped_inputs = tuple(shaped_inputs)
            
            outputs = func(shaped_inputs)
            
            if isinstance(outputs, torch.Tensor):
                return outputs.flatten()
            else:
                return torch.cat([out.flatten() for out in outputs])
        except RuntimeError as e:
            if "Expected tensor for argument #1 'indices' to have" in str(e):
                # This is likely an embedding function issue with BFloat16 inputs
                if debug:
                    print("Converting to float32 for embedding function compatibility.")
                
                # Convert inputs to float32 for embedding compatibility
                if isinstance(inputs, torch.Tensor):
                    shaped_inputs = flat_inputs.reshape(input_shape).float()
                else:
                    shaped_inputs = []
                    offset = 0
                    for i, shape in enumerate(input_shape):
                        size = torch.prod(torch.tensor(shape)).item()
                        shaped_inputs.append(flat_inputs[offset:offset+size].reshape(shape).float())
                        offset += size
                    shaped_inputs = tuple(shaped_inputs)
                
                outputs = func(shaped_inputs)
                
                if isinstance(outputs, torch.Tensor):
                    return outputs.flatten()
                else:
                    return torch.cat([out.flatten() for out in outputs])
            else:
                # Re-raise if it's not the specific error we're handling
                raise
    
    # Create flattened inputs in original dtype
    if isinstance(inputs, torch.Tensor):
        flat_inputs = inputs.flatten().requires_grad_(True)
    else:
        flat_inputs = torch.cat([inp.flatten() for inp in inputs]).requires_grad_(True)
    
    # Define a function that computes JVP for a single vector with memory cleanup
    def jvp_single(v):
        # Convert to model dtype if needed
        if needs_conversion:
            v_model = v.to(original_dtype)
        else:
            v_model = v
        
        # Compute JVP
        with torch.no_grad():  # ← IMPORTANT: Prevent gradient accumulation outside JVP
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        # Compute JVP with explicit cleanup
        try:
            # Perform the actual JVP operation (this needs gradients)
            _, result = jvp(flattened_func, (flat_inputs,), (v_model,))
            
            # Convert result to computation precision if needed
            if needs_conversion:
                result = result.float()
                
            # Detach result to prevent gradient graph buildup
            result = result.detach()  # ← CRITICAL: Break gradient graph here
            
            return result
        except Exception as e:
            # Clean up on error
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            raise e
    
    # Define a function that computes VJP for a single vector with memory cleanup
    def vjp_single(u):
        # Convert to model dtype if needed
        if needs_conversion:
            u_model = u.to(original_dtype)
        else:
            u_model = u
            
        # Compute VJP
        with torch.no_grad():  # ← IMPORTANT: Prevent gradient accumulation outside VJP
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        # Compute VJP with explicit cleanup
        try:
            # Create VJP function
            _, vjp_fn = vjp(flattened_func, flat_inputs)
            
            # Apply VJP function to get result
            result = vjp_fn(u_model)
            
            # Extract first element if tuple
            if isinstance(result, tuple):
                result = result[0]
                
            # Convert result to computation precision if needed
            if needs_conversion:
                result = result.float()
                
            # Detach result to prevent gradient graph buildup
            result = result.detach()  # ← CRITICAL: Break gradient graph here
            
            return result
        except Exception as e:
            # Clean up on error
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            raise e
    
    # Vectorized versions that operate on matrices (each column is a vector)
    # Using explicit no_grad to prevent gradient buildup
    def jvp_matrix(matrix):
        """Apply JVP to each column of the matrix and return results as a matrix."""
        # matrix shape: [input_dim, num_cols]
        with torch.no_grad():  # ← IMPORTANT: Prevent gradient accumulation around vmap
            # Transpose to get [num_cols, input_dim] for vmap's expected in_dims=0
            matrix_t = matrix.T
            
            # Use vmap to apply jvp_single to each row of matrix_t
            # This effectively applies jvp to each column of the original matrix
            results_t = vmap(jvp_single)(matrix_t)
            
            # results_t shape: [num_cols, output_dim]
            # Transpose back to get [output_dim, num_cols]
            result = results_t.T
            
            # Force detach to ensure no gradient information is retained
            result = result.detach()  # ← Extra safety: Ensure result is detached
            
            # Clean up intermediate tensors
            del matrix_t, results_t  # ← Delete intermediate tensors explicitly
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return result
    
    def vjp_matrix(matrix):
        """Apply VJP to each column of the matrix and return results as a matrix."""
        # matrix shape: [output_dim, num_cols]
        with torch.no_grad():  # ← IMPORTANT: Prevent gradient accumulation around vmap
            # Transpose to get [num_cols, output_dim] for vmap's expected in_dims=0
            matrix_t = matrix.T
            
            # Use vmap to apply vjp_single to each row of matrix_t
            # This effectively applies vjp to each column of the original matrix
            results_t = vmap(vjp_single)(matrix_t)
            
            # results_t shape: [num_cols, input_dim]
            # Transpose back to get [input_dim, num_cols]
            result = results_t.T
            
            # Force detach to ensure no gradient information is retained
            result = result.detach()  # ← Extra safety: Ensure result is detached
            
            # Clean up intermediate tensors
            del matrix_t, results_t  # ← Delete intermediate tensors explicitly
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return result
    
    if debug:
        print("Created vectorized matrix-vector product functions with vmap")
    
    # Clean up input tensors we don't need anymore
    if not isinstance(inputs, torch.Tensor):
        for inp in inputs:
            # Release any non-essential references
            if hasattr(inp, '_grad_fn') and inp._grad_fn is not None:
                inp._grad_fn = None  # ← Break potential reference cycles in gradient graph
    
    # Final memory cleanup
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return jvp_matrix, vjp_matrix


# Example usage of the high accuracy version for better lower singular vectors
def get_accurate_svd_for_token(func, inputs, token_idx=0, num_singular_vectors=10):
    """Helper function to get accurate SVD for a specific token."""
    import gc

    assert isinstance(inputs, torch.Tensor) and len(inputs.shape) == 3, \
        "Per-token analysis requires inputs of shape [batch, seq_len, emb_dim]"

    # Create token-specific function
    def token_specific_func(token_emb):
        modified_inputs = inputs.clone()
        modified_inputs[0, token_idx, :] = token_emb
        return func(modified_inputs)

    # Extract token embedding
    token_emb = inputs[0, token_idx, :].clone().requires_grad_(True)

    # Memory cleanup before computation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute high-accuracy SVD
    U, S, V = high_accuracy_svd(
        token_specific_func,
        token_emb,
        num_singular_vectors=num_singular_vectors,
        num_iter=8,  # More power iterations
        oversampling=2*num_singular_vectors,  # More oversampling
        debug=True
    )

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return U, S, V

#####@title Vectorized SVD
def jacobian_svd_vectorized(func, inputs, num_singular_vectors=5, create_graph=False, strict=False,
                          vectorize=False, strategy='auto', disable_flash_attn=True, debug=False,
                          max_memory_gb=8, gpu_memory_fraction=0.8, per_token=False,
                          return_numpy=False, **svd_kwargs):
    """
    Optimized version of jacobian_svd using vectorized operations with vmap.

    This function mimics the interface of torch.autograd.functional.jacobian but returns
    singular value decomposition results instead of the full Jacobian matrix, and uses
    vectorized operations for faster computation.

    Args:
        func: A Python function that takes Tensor inputs and returns
              a Tensor or tuple of Tensors
        inputs: Tensor or tuple of Tensors that are inputs to `func`
        num_singular_vectors: Number of top singular vectors to compute (default: 5)
        create_graph: If True, the Jacobian will be computed in a differentiable manner (not implemented)
        strict: If True, an error will be raised when we detect that there exists an input
               such that all the outputs are independent of it (not implemented)
        vectorize: This flag is experimental, use at your own risk (ignored)
        strategy: 'auto', 'full', or 'randomized'
                 - 'auto': automatically choose based on problem size
                 - 'full': always compute full Jacobian first (memory intensive)
                 - 'randomized': always use randomized method
        disable_flash_attn: Whether to disable Flash Attention for JVP/VJP compatibility (default: True)
        debug: Enable debugging output (default: False)
        max_memory_gb: Maximum memory to use for full Jacobian (default: 8GB)
        gpu_memory_fraction: Fraction of free GPU memory to use (default: 0.8)
        per_token: If True, return separate SVD for each token position (default: False)
        return_numpy: If True, return numpy arrays instead of tensors (saves GPU memory)
        **svd_kwargs: Additional arguments passed to the SVD computation
                     (num_iter, oversampling, debug, etc.)

    Returns:
        JacobianSVD: A named tuple containing:
            - U: Left singular vectors (output space) [output_dim x num_singular_vectors]
                 OR [num_tokens, output_dim, num_singular_vectors] if per_token=True
            - S: Singular values [num_singular_vectors]
                 OR [num_tokens, num_singular_vectors] if per_token=True
            - V: Right singular vectors (input space) [input_dim x num_singular_vectors]
                 OR [num_tokens, token_dim, num_singular_vectors] if per_token=True
            - input_dim: Flattened input dimension
            - output_dim: Flattened output dimension
            - num_tokens: Number of tokens (only if per_token=True)
    """
    from collections import namedtuple
    import gc

    # Define return type (updated for per-token support)
    if per_token:
        JacobianSVD = namedtuple('JacobianSVD', ['U', 'S', 'V', 'input_dim', 'output_dim', 'num_tokens'])
    else:
        JacobianSVD = namedtuple('JacobianSVD', ['U', 'S', 'V', 'input_dim', 'output_dim'])

    # Handle create_graph and strict warnings
    if create_graph:
        print("Warning: create_graph=True is not yet implemented for jacobian_svd_vectorized")
    if strict:
        print("Warning: strict=True is not yet implemented for jacobian_svd_vectorized")

    # Safety check for reasonable problem size
    try:
        device, dtype, input_dim, output_dim, input_shape, output_shape = _setup_dimensions(inputs, func, debug=debug)
        if debug:
            print(f"Problem dimensions: {output_dim} × {input_dim}")
            print(f"Input shape: {input_shape}")
            print(f"Output shape: {output_shape}")
            print(f"Input dtype: {dtype}")
    except Exception as e:
        print(f"Error during dimension setup: {e}")
        raise

    # Create context manager for Flash Attention handling
    context_manager = disable_flash_attention() if disable_flash_attn else nullcontext()

    # Per-token analysis for LLMs
    with context_manager:
        if per_token:
            if debug:
                print("Performing per-token Jacobian analysis with vectorization...")

            # Check if input has the expected shape for token-level analysis
            if isinstance(inputs, torch.Tensor) and len(input_shape) == 3:
                # Shape: [batch, num_tokens, token_dim]
                batch_size, num_tokens, token_dim = input_shape
                if batch_size != 1:
                    raise ValueError(f"Per-token analysis requires batch_size=1, got {batch_size}")

                if debug:
                    print(f"Detected {num_tokens} tokens, each with {token_dim} dimensions")

                # Use the specialized per-token implementation
                U_tokens, S_tokens, V_tokens = randomized_svd_jacobian_per_token(
                    func, inputs,
                    num_singular_vectors=num_singular_vectors,
                    num_iter=svd_kwargs.get('num_iter', 4),
                    oversampling=svd_kwargs.get('oversampling', 10),
                    debug=debug
                )

                # Convert to numpy if requested
                if return_numpy:
                    U_tokens = U_tokens.detach().cpu().numpy()
                    S_tokens = S_tokens.detach().cpu().numpy()
                    V_tokens = V_tokens.detach().cpu().numpy()

                # Clean up memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return JacobianSVD(U=U_tokens, S=S_tokens, V=V_tokens,
                                input_dim=input_dim, output_dim=output_dim, num_tokens=num_tokens)

            else:
                raise ValueError(f"Per-token analysis requires 3D input tensor [batch, tokens, dim], got shape {input_shape}")

        # Standard analysis (single Jacobian SVD)
        else:
            # Memory safety check for full strategy
            jacobian_memory_gb = (input_dim * output_dim * 4) / (1024**3)  # 4 bytes per float32

            # GPU memory check if using CUDA
            if torch.cuda.is_available() and device.type == 'cuda':
                gpu_info = get_gpu_memory_info()
                if debug:
                    print(f"GPU Memory - Total: {gpu_info['total_gb']:.1f}GB, "
                        f"Free: {gpu_info['free_gb']:.1f}GB")
                    print(f"Full Jacobian would require ~{jacobian_memory_gb:.2f}GB")

                # Use GPU memory limit if more restrictive
                gpu_available = gpu_info['free_gb'] * gpu_memory_fraction
                effective_memory_limit = min(max_memory_gb, gpu_available)
            else:
                effective_memory_limit = max_memory_gb

            # Choose strategy with safety checks
            if strategy == 'auto':
                if jacobian_memory_gb > effective_memory_limit:
                    strategy_chosen = 'randomized'
                    if debug:
                        print(f"Auto-choosing randomized (memory: {jacobian_memory_gb:.2f}GB > {effective_memory_limit:.1f}GB)")
                else:
                    strategy_chosen = 'full'
                    if debug:
                        print(f"Auto-choosing full SVD (memory: {jacobian_memory_gb:.2f}GB <= {effective_memory_limit:.1f}GB)")
            else:
                strategy_chosen = strategy

            # Safety check for randomized method
            if strategy_chosen == 'randomized' and num_singular_vectors >= min(input_dim, output_dim):
                print(f"Warning: num_singular_vectors ({num_singular_vectors}) >= min(input_dim, output_dim) ({min(input_dim, output_dim)})")
                num_singular_vectors_safe = min(num_singular_vectors, min(input_dim, output_dim) - 1)
                print(f"Reducing to {num_singular_vectors_safe}")
            else:
                num_singular_vectors_safe = num_singular_vectors

            # Compute SVD based on strategy
            try:
                if strategy_chosen == 'full':
                    if debug:
                        print("Computing full SVD...")
                    U, S, V = compute_ground_truth_svd(func, inputs, num_singular_vectors_safe)

                elif strategy_chosen == 'randomized':
                    if debug:
                        print("Computing randomized SVD with vectorization...")

                    # Get default SVD parameters
                    default_svd_params = {
                        'num_iter': 4,
                        'oversampling': min(10, min(input_dim, output_dim) // 2),
                        'debug': debug,
                        'stabilize': True
                    }
                    default_svd_params.update(svd_kwargs)

                    U, S, V = randomized_svd_jacobian_vectorized(
                        func, inputs,
                        num_singular_vectors=num_singular_vectors_safe,
                        **default_svd_params
                    )
                else:
                    raise ValueError(f"Unknown strategy: {strategy_chosen}. Must be 'auto', 'full', or 'randomized'")

                # Convert to numpy if requested
                if return_numpy:
                    U = U.detach().cpu().numpy()
                    S = S.detach().cpu().numpy()
                    V = V.detach().cpu().numpy()

                # Clean up memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return JacobianSVD(U=U, S=S, V=V, input_dim=input_dim, output_dim=output_dim)

            except Exception as e:
                if debug:
                    print(f"Error during SVD computation: {e}")
                if strategy_chosen == 'full':
                    print(f"Full SVD failed ({e}), falling back to randomized method")
                    try:
                        default_svd_params = {
                            'num_iter': 4,
                            'oversampling': min(10, min(input_dim, output_dim) // 2),
                            'debug': debug,
                            'stabilize': True
                        }
                        default_svd_params.update(svd_kwargs)

                        U, S, V = randomized_svd_jacobian_vectorized(
                            func, inputs,
                            num_singular_vectors=num_singular_vectors_safe,
                            **default_svd_params
                        )

                        if return_numpy:
                            U = U.detach().cpu().numpy()
                            S = S.detach().cpu().numpy()
                            V = V.detach().cpu().numpy()

                        return JacobianSVD(U=U, S=S, V=V, input_dim=input_dim, output_dim=output_dim)
                    except Exception as e2:
                        print(f"Randomized SVD also failed: {e2}")
                        raise
                # else:
                #     raise

import torch
import torch.nn.functional as F
from torch.func import jvp, vjp, vmap
import numpy as np
import math
from contextlib import contextmanager
import time


@contextmanager
def disable_flash_attention():
    """Context manager to temporarily disable Flash Attention for JVP/VJP compatibility."""
    # Store original states
    original_flash = torch.backends.cuda.flash_sdp_enabled()
    original_mem_efficient = torch.backends.cuda.mem_efficient_sdp_enabled()
    original_math = torch.backends.cuda.math_sdp_enabled()

    try:
        # Disable Flash Attention and Memory Efficient Attention, enable Math Attention
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        yield
    finally:
        # Restore original states
        torch.backends.cuda.enable_flash_sdp(original_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(original_mem_efficient)
        torch.backends.cuda.enable_math_sdp(original_math)


def randomized_svd_jacobian_vectorized(func, inputs, num_singular_vectors=5, num_iter=4,
                                     oversampling=10, debug=False, stabilize=True):
    """
    Vectorized randomized SVD for Jacobian matrices using vmap for JVP/VJP operations.

    This implementation uses torch.func.vmap to vectorize the computation of JVP and VJP
    operations across multiple vectors at once, which can significantly improve performance.

    Args:
        func: Function whose Jacobian we want to analyze
        inputs: Input tensor(s) to the function
        num_singular_vectors: Number of top singular vectors to compute
        num_iter: Number of power iterations for accuracy
        oversampling: Extra random vectors for stability
        debug: Print debugging information
        stabilize: Use numerical stabilization techniques

    Returns:
        U: Left singular vectors (output space)
        S: Singular values
        V: Right singular vectors (input space)
    """
    import gc
    # Use disable_flash_attention context manager for JVP/VJP compatibility
    with disable_flash_attention():
        # Setup and dimension calculation
        device, dtype, input_dim, output_dim, input_shape, output_shape = _setup_dimensions(inputs, func, debug)

        k = num_singular_vectors

        # Adaptive parameters for small problems
        if min(input_dim, output_dim) < 20:
            # For very small problems, use more aggressive settings
            oversampling = max(oversampling, min(input_dim, output_dim) - k)
            num_iter = max(num_iter, 6)
            if debug:
                print(f"Small problem detected: using oversampling={oversampling}, num_iter={num_iter}")

        l = min(k + oversampling, min(input_dim, output_dim))

        if debug:
            print(f"Input dim: {input_dim}, Output dim: {output_dim}, k={k}, l={l}")

        # Create vectorized matrix-vector product functions
        jvp_vmap, vjp_vmap = _create_vectorized_matrix_vector_functions(
            func, inputs, input_shape, output_shape, input_dim, output_dim, debug
        )

        # Randomized SVD Algorithm with vmap optimization

        # Step 1: Generate random test matrix
        Omega = torch.randn(input_dim, l, device=device, dtype=dtype)
        if stabilize:
            Omega, _ = _safe_qr_decomposition(Omega, debug)
        del _

        # Step 2: Form Y = A * Omega using vectorized JVP
        # Instead of a loop, we apply jvp to all columns of Omega at once
        Y = jvp_vmap(Omega)  # Shape: [output_dim, l]

        # Step 3: Power iterations for improved accuracy
        for iteration in range(num_iter):
            # Orthogonalize Y
            if stabilize:
                Y, _ = _safe_qr_decomposition(Y, debug)
            del _

            # Z = A^T * Y using vectorized VJP
            Z = vjp_vmap(Y)  # Shape: [input_dim, l]

            # Orthogonalize Z
            if stabilize:
                Z, _ = _safe_qr_decomposition(Z, debug)
                del _

            # Y = A * Z using vectorized JVP
            Y = jvp_vmap(Z)  # Shape: [output_dim, l]
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 4: QR decomposition of Y
        Q, R = _safe_qr_decomposition(Y, debug)
        del R
        Q = Q[:, :k]  # Keep only first k columns

        # Step 5: Form the small matrix B = Q^T * A * Omega_k
        # Create a fresh set of orthogonal vectors for the final projection
        if l > k:
            # Use the first k columns of the original Omega, but ensure they're orthogonal
            Omega_k = Omega[:, :k]
            Omega_k, _ = torch.linalg.qr(Omega_k)
            del _
        else:
            Omega_k = Omega

        # Apply JVP to all columns of Omega_k at once
        Y_omega = jvp_vmap(Omega_k)  # Shape: [output_dim, k]

        # Project onto the subspace spanned by Q
        B = Q.T @ Y_omega  # Shape: [k, k]
        del Y_omega

        # Step 6: SVD of the small matrix B
        U_tilde, S, Vt_tilde = _safe_svd(B, debug)
        del B

        # Step 7: Recover the singular vectors
        U = Q @ U_tilde  # Left singular vectors
        del Q, U_tilde

        # Right singular vectors: V = Omega_k * V_tilde^T
        V = Omega_k @ Vt_tilde.T
        del Omega_k, Vt_tilde

        # Normalize right singular vectors
        V_norms = torch.norm(V, dim=0, keepdim=True)
        mask = V_norms > 1e-10
        V[:, mask.squeeze()] = V[:, mask.squeeze()] / V_norms[:, mask.squeeze()]
        del V_norms

        
        U_clean, S_clean, V_clean = U.clone().detach(), S.clone().detach(), V.clone().detach()
        del U, S, V

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if debug:
            print(f"Final shapes - U: {U_clean.shape}, S: {S_clean.shape}, V: {V_clean.shape}")
            print(f"Singular values: {S_clean}")

        return U_clean, S_clean, V_clean


def _setup_dimensions(inputs, func, debug=False):
    """Setup device, dtype, and dimension information with automatic dtype handling."""

    if isinstance(inputs, torch.Tensor):
        inputs = inputs.requires_grad_(True)
        input_shape = inputs.shape
        input_dim = inputs.numel()
        device = inputs.device
        dtype = inputs.dtype
    else:
        inputs = [x.requires_grad_(True) for x in inputs]
        input_shape = [x.shape for x in inputs]
        input_dim = sum(x.numel() for x in inputs)
        device = inputs[0].device
        dtype = inputs[0].dtype

    # Get output dimensions using original inputs (don't convert yet)
    # Make a clone of inputs to avoid modifying the original
    if isinstance(inputs, torch.Tensor):
        input_clone = inputs.clone()
    else:
        input_clone = [x.clone() for x in inputs]

    try:
        outputs = func(input_clone)
        if isinstance(outputs, torch.Tensor):
            output_shape = outputs.shape
            output_dim = outputs.numel()
        else:
            output_shape = [out.shape for out in outputs]
            output_dim = sum(out.numel() for out in outputs)
    except RuntimeError as e:
        if "Expected tensor for argument #1 'indices' to have" in str(e):
            # This is likely an embedding function issue with BFloat16 inputs
            if debug:
                print("Detected potential embedding function issue with BFloat16 inputs.")
                print("Converting to float32 for function evaluation only.")

            # Convert inputs to float32 for the test run
            if isinstance(inputs, torch.Tensor):
                input_clone = inputs.float()
            else:
                input_clone = [x.float() for x in inputs]

            # Try again with float32
            outputs = func(input_clone)
            if isinstance(outputs, torch.Tensor):
                output_shape = outputs.shape
                output_dim = outputs.numel()
            else:
                output_shape = [out.shape for out in outputs]
                output_dim = sum(out.numel() for out in outputs)
        else:
            # Re-raise if it's not the specific error we're handling
            raise

    if debug:
        print(f"Input shape: {input_shape}, dim: {input_dim}")
        print(f"Output shape: {output_shape}, dim: {output_dim}")
        print(f"Using dtype: {dtype}")

    return device, dtype, input_dim, output_dim, input_shape, output_shape


def _create_vectorized_matrix_vector_functions(func, inputs, input_shape, output_shape,
                                            input_dim, output_dim, debug=False):
    """
    Create vectorized JVP and VJP functions using vmap.

    This function creates JVP and VJP functions that operate on entire matrices at once,
    rather than processing each column individually.
    """

    # Keep original dtype for model compatibility
    original_dtype = inputs.dtype if isinstance(inputs, torch.Tensor) else inputs[0].dtype
    needs_conversion = original_dtype in [torch.bfloat16, torch.float16]

    if needs_conversion and debug:
        print(f"Will convert to float32 only for linear algebra operations (model uses: {original_dtype})")

    # Create flattened function that matches jacobian() exactly
    def flattened_func(flat_inputs):
        try:
            if isinstance(inputs, torch.Tensor):
                shaped_inputs = flat_inputs.reshape(input_shape)
            else:
                shaped_inputs = []
                offset = 0
                for i, shape in enumerate(input_shape):
                    size = torch.prod(torch.tensor(shape)).item()
                    shaped_inputs.append(flat_inputs[offset:offset+size].reshape(shape))
                    offset += size
                shaped_inputs = tuple(shaped_inputs)

            outputs = func(shaped_inputs)

            if isinstance(outputs, torch.Tensor):
                return outputs.flatten()
            else:
                return torch.cat([out.flatten() for out in outputs])
        except RuntimeError as e:
            if "Expected tensor for argument #1 'indices' to have" in str(e):
                # This is likely an embedding function issue with BFloat16 inputs
                if debug:
                    print("Converting to float32 for embedding function compatibility.")

                # Convert inputs to float32 for embedding compatibility
                if isinstance(inputs, torch.Tensor):
                    shaped_inputs = flat_inputs.reshape(input_shape).float()
                else:
                    shaped_inputs = []
                    offset = 0
                    for i, shape in enumerate(input_shape):
                        size = torch.prod(torch.tensor(shape)).item()
                        shaped_inputs.append(flat_inputs[offset:offset+size].reshape(shape).float())
                        offset += size
                    shaped_inputs = tuple(shaped_inputs)

                outputs = func(shaped_inputs)

                if isinstance(outputs, torch.Tensor):
                    return outputs.flatten()
                else:
                    return torch.cat([out.flatten() for out in outputs])
            else:
                # Re-raise if it's not the specific error we're handling
                raise

    # Create flattened inputs in original dtype
    if isinstance(inputs, torch.Tensor):
        flat_inputs = inputs.flatten().requires_grad_(True)
    else:
        flat_inputs = torch.cat([inp.flatten() for inp in inputs]).requires_grad_(True)

    # Define a function that computes JVP for a single vector
    def jvp_single(v):
        if needs_conversion:
            v_model = v.to(original_dtype)
        else:
            v_model = v

        _, result = jvp(flattened_func, (flat_inputs,), (v_model,))

        if needs_conversion:
            result = result.float()
        return result

    # Define a function that computes VJP for a single vector
    def vjp_single(u):
        if needs_conversion:
            u_model = u.to(original_dtype)
        else:
            u_model = u

        _, vjp_fn = vjp(flattened_func, flat_inputs)
        result = vjp_fn(u_model)

        if isinstance(result, tuple):
            result = result[0]

        if needs_conversion:
            result = result.float()
        return result

    # Vectorized versions that operate on matrices (each column is a vector)
    def jvp_matrix(matrix):
        """Apply JVP to each column of the matrix and return results as a matrix."""
        # matrix shape: [input_dim, num_cols]
        # Transpose to get [num_cols, input_dim] for vmap's expected in_dims=0
        matrix_t = matrix.T

        # Use vmap to apply jvp_single to each row of matrix_t
        # This effectively applies jvp to each column of the original matrix
        results_t = vmap(jvp_single)(matrix_t)

        # results_t shape: [num_cols, output_dim]
        # Transpose back to get [output_dim, num_cols]
        return results_t.T

    def vjp_matrix(matrix):
        """Apply VJP to each column of the matrix and return results as a matrix."""
        # matrix shape: [output_dim, num_cols]
        # Transpose to get [num_cols, output_dim] for vmap's expected in_dims=0
        matrix_t = matrix.T

        # Use vmap to apply vjp_single to each row of matrix_t
        # This effectively applies vjp to each column of the original matrix
        results_t = vmap(vjp_single)(matrix_t)

        # results_t shape: [num_cols, input_dim]
        # Transpose back to get [input_dim, num_cols]
        return results_t.T

    if debug:
        print("Created vectorized matrix-vector product functions with vmap")

    return jvp_matrix, vjp_matrix


def _safe_qr_decomposition(matrix, debug=False):
    """Perform QR decomposition with automatic dtype conversion for CUDA compatibility."""
    original_dtype = matrix.dtype
    needs_conversion = original_dtype in [torch.bfloat16, torch.float16]

    if needs_conversion:
        if debug:
            print(f"Converting to float32 for QR decomposition (original: {original_dtype})")
        matrix_float = matrix.float()
        Q, R = torch.linalg.qr(matrix_float)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # Convert back to original dtype if needed (though we usually want float32 for subsequent ops)
        return Q, R
    else:
        return torch.linalg.qr(matrix)


def _safe_svd(matrix, debug=False):
    """Perform SVD with automatic dtype conversion for CUDA compatibility."""
    original_dtype = matrix.dtype
    needs_conversion = original_dtype in [torch.bfloat16, torch.float16]

    if needs_conversion:
        # if debug:
        #     print(f"Converting to float32 for SVD (original: {original_dtype})")
        # matrix_float = matrix.float()

        if debug:
            print(f"Converting to float64 for SVD (original: {original_dtype})")
        # Convert to double precision for better numerical stability
        matrix_double = matrix.double()
        return torch.linalg.svd(matrix_double, full_matrices=False)
    else:
        return torch.linalg.svd(matrix, full_matrices=False)


def compute_ground_truth_svd(func, inputs, num_singular_vectors=None):
    """
    Compute ground truth SVD using torch.autograd.functional.jacobian.
    WARNING: Only use for small problems!
    Handles dtype conversion automatically.
    """
    from torch.autograd.functional import jacobian

    print("Computing full Jacobian for ground truth...")

    # Handle dtype conversion
    original_dtype = inputs.dtype if isinstance(inputs, torch.Tensor) else inputs[0].dtype
    needs_conversion = original_dtype in [torch.bfloat16, torch.float16]

    if needs_conversion:
        print(f"Converting from {original_dtype} to float32 for ground truth computation")
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.float()
        else:
            inputs = [x.float() for x in inputs]

    # Create the same flattened function used in our randomized approach
    if isinstance(inputs, torch.Tensor):
        input_shape = inputs.shape
        input_dim = inputs.numel()
    else:
        input_shape = [x.shape for x in inputs]
        input_dim = sum(x.numel() for x in inputs)

    def flattened_func(flat_inputs):
        # Convert to original dtype for model execution
        if needs_conversion:
            flat_inputs = flat_inputs.to(original_dtype)

        if isinstance(inputs, torch.Tensor):
            shaped_inputs = flat_inputs.reshape(input_shape)
        else:
            shaped_inputs = []
            offset = 0
            for i, shape in enumerate(input_shape):
                size = torch.prod(torch.tensor(shape)).item()
                shaped_inputs.append(flat_inputs[offset:offset+size].reshape(shape))
                offset += size
            shaped_inputs = tuple(shaped_inputs)

        outputs = func(shaped_inputs)
        if isinstance(outputs, torch.Tensor):
            result = outputs.flatten()
        else:
            result = torch.cat([out.flatten() for out in outputs])

        # Convert back to float32 for linear algebra
        if needs_conversion:
            result = result.float()

        return result

    # Flatten inputs exactly as in randomized version
    if isinstance(inputs, torch.Tensor):
        flat_inputs = inputs.flatten().requires_grad_(True)
    else:
        flat_inputs = torch.cat([inp.flatten() for inp in inputs]).requires_grad_(True)

    # Ensure we're using float32 for linear algebra
    if needs_conversion:
        flat_inputs = flat_inputs.float().requires_grad_(True)

    # Compute Jacobian
    jac = jacobian(flattened_func, flat_inputs, vectorize=True)
    print(f"Full Jacobian shape: {jac.shape}")

    # Compute SVD
    U_full, S_full, Vt_full = torch.linalg.svd(jac, full_matrices=False)

    # Return only top k if specified
    if num_singular_vectors is not None:
        k = min(num_singular_vectors, len(S_full))
        return U_full[:, :k], S_full[:k], Vt_full[:k].T
    else:
        return U_full, S_full, Vt_full.T


def validate_randomized_svd(func, inputs, num_singular_vectors=3, test_ground_truth=True):
    """
    Comprehensive validation of the randomized SVD implementation.
    Compares both standard and vectorized implementations.
    """
    print("="*60)
    print("VALIDATING RANDOMIZED SVD IMPLEMENTATIONS")
    print("="*60)

    # Get problem dimensions
    if isinstance(inputs, torch.Tensor):
        input_dim = inputs.numel()
    else:
        input_dim = sum(x.numel() for x in inputs)

    outputs = func(inputs)
    if isinstance(outputs, torch.Tensor):
        output_dim = outputs.numel()
    else:
        output_dim = sum(out.numel() for out in outputs)

    print(f"Problem size: {output_dim} × {input_dim}")

    # Use the original randomized_svd_jacobian_improved from this file
    # (Instead of importing from paste module which doesn't exist)

    # Test original implementation (the function passed in as an argument)
    print("\n1. Computing with original implementation...")
    start_time = time.time()
    U_orig, S_orig, V_orig = randomized_svd_jacobian_improved(
        func, inputs,
        num_singular_vectors=num_singular_vectors,
        num_iter=4,
        oversampling=10,
        debug=False,
        stabilize=True
    )
    orig_time = time.time() - start_time
    print(f"Original implementation time: {orig_time:.4f} seconds")
    print(f"Original SVD singular values: {S_orig.detach().numpy()}")

    # Test vectorized implementation
    print("\n2. Computing with vectorized implementation...")
    start_time = time.time()
    U_vect, S_vect, V_vect = randomized_svd_jacobian_vectorized(
        func, inputs,
        num_singular_vectors=num_singular_vectors,
        num_iter=4,
        oversampling=10,
        debug=False,
        stabilize=True
    )
    vect_time = time.time() - start_time
    print(f"Vectorized implementation time: {vect_time:.4f} seconds")
    print(f"Vectorized SVD singular values: {S_vect.detach().numpy()}")

    # Compare implementations
    print("\n3. COMPARING IMPLEMENTATIONS:")
    print("-" * 40)
    speedup = orig_time / vect_time
    print(f"Speedup factor: {speedup:.2f}x faster with vectorization")

    # Singular value differences
    s_diff = torch.abs(S_orig - S_vect) / (S_orig + 1e-10)
    print(f"Relative singular value differences: {s_diff.detach().numpy()}")
    print(f"Max relative difference: {s_diff.max().item():.6f}")

    # Test against ground truth if problem is small enough
    if test_ground_truth and input_dim * output_dim < 50000:
        print("\n4. Computing ground truth...")
        try:
            start_time = time.time()
            U_true, S_true, V_true = compute_ground_truth_svd(func, inputs, num_singular_vectors)
            true_time = time.time() - start_time
            print(f"Ground truth time: {true_time:.4f} seconds")
            print(f"Ground truth singular values: {S_true.detach().numpy()}")

            print("\n5. ACCURACY ANALYSIS:")
            print("-" * 40)

            # Original implementation errors
            s_error_orig = torch.abs(S_orig - S_true) / (S_true + 1e-10)
            print(f"Original implementation - Max relative error: {s_error_orig.max().item():.6f}")

            # Vectorized implementation errors
            s_error_vect = torch.abs(S_vect - S_true) / (S_true + 1e-10)
            print(f"Vectorized implementation - Max relative error: {s_error_vect.max().item():.6f}")

            # Overall assessment
            if s_error_vect.max().item() < 0.01:
                print("\n✅ EXCELLENT: Vectorized implementation maintains high accuracy!")
            elif s_error_vect.max().item() < 0.1:
                print("\n✅ GOOD: Vectorized implementation has reasonable accuracy")
            else:
                print("\n⚠️  CAUTION: Vectorized implementation may have accuracy issues")

        except Exception as e:
            print(f"\n⚠️  Ground truth computation failed: {e}")

    else:
        print("\n⚠️  Skipping ground truth (problem too large or disabled)")

    print("\n6. PERFORMANCE SUMMARY:")
    print("-" * 40)
    print(f"• Original implementation: {orig_time:.4f} seconds")
    print(f"• Vectorized implementation: {vect_time:.4f} seconds")
    print(f"• Speedup: {speedup:.2f}x")

    return U_vect, S_vect, V_vect


def randomized_svd_jacobian_per_token(func, inputs, num_singular_vectors=5, num_iter=4,
                                    oversampling=10, debug=False):
    """
    Compute SVD of Jacobian for each token position in a batched sequence.
    Uses vectorized operations for efficiency.

    Args:
        func: Function whose Jacobian we want to analyze (expected to take [batch, seq_len, emb_dim])
        inputs: Input tensor with shape [batch, seq_len, emb_dim]
        num_singular_vectors: Number of top singular vectors to compute per token
        num_iter: Number of power iterations for accuracy
        oversampling: Extra random vectors for stability
        debug: Print debugging information

    Returns:
        U_per_token: Left singular vectors for each token [seq_len, output_dim, k]
        S_per_token: Singular values for each token [seq_len, k]
        V_per_token: Right singular vectors for each token [seq_len, emb_dim, k]
    """
    # Use disable_flash_attention context manager for JVP/VJP compatibility
    import gc
    with disable_flash_attention():
        assert isinstance(inputs, torch.Tensor) and len(inputs.shape) == 3, \
            "Per-token analysis requires inputs of shape [batch, seq_len, emb_dim]"

        batch_size, seq_len, emb_dim = inputs.shape
        assert batch_size == 1, "Per-token analysis currently supports batch_size=1 only"

        if debug:
            print(f"Computing per-token Jacobian SVD for sequence length {seq_len}")
            print(f"Each token has embedding dimension {emb_dim}")

        # Initialize storage for results
        device = inputs.device
        dtype = inputs.dtype

        # First, get output dimensions by running the function once
        outputs = func(inputs)
        if isinstance(outputs, torch.Tensor):
            output_shape = outputs.shape
            output_dim = outputs.numel()
        else:
            raise ValueError("Per-token analysis expects tensor output from func")

        if debug:
            print(f"Output shape: {output_shape}, output_dim: {output_dim}")

        # Prepare containers for results
        U_per_token = []
        S_per_token = []
        V_per_token = []

        # Define function that will be used for each token
        def token_func(token_idx, token_emb):
            # Create a copy of the inputs with the specified token replaced
            modified_inputs = inputs.clone()
            modified_inputs[0, token_idx, :] = token_emb
            return func(modified_inputs)

        # For each token position, compute the SVD of its Jacobian
        for token_idx in range(seq_len):
            if debug:
                print(f"Processing token {token_idx}/{seq_len}...")

            # Function that only varies this specific token's embedding
            def token_specific_func(token_emb):
                return token_func(token_idx, token_emb)

            # Extract the token embedding
            token_emb = inputs[0, token_idx, :].clone().requires_grad_(True)

            # Optional: print memory usage for debugging
            if debug and torch.cuda.is_available():
                print(f"GPU memory before token {token_idx}: "
                      f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated, "
                      f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB reserved")

                # Clear garbage 
            # Compute SVD for this token
            U, S, V = randomized_svd_jacobian_vectorized(
                token_specific_func, token_emb,
                num_singular_vectors=num_singular_vectors,
                num_iter=num_iter,
                oversampling=oversampling,
                debug=debug if token_idx == 0 else False,  # Only debug first token
                stabilize=True
            )

            U_per_token.append(U.detach().cpu().numpy())
            S_per_token.append(S.detach().cpu().numpy())
            V_per_token.append(V.detach().cpu().numpy())
            del token_emb, token_specific_func
            del U, S, V
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete

            # Optional: print memory usage for debugging
            if debug and torch.cuda.is_available():
                print(f"GPU memory after token {token_idx}: "
                      f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated, "
                      f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB reserved")


        # Stack results
        # U_stacked = torch.stack(U_per_token, dim=0)  # [seq_len, output_dim, k]
        # S_stacked = torch.stack(S_per_token, dim=0)  # [seq_len, k]
        # V_stacked = torch.stack(V_per_token, dim=0)  # [seq_len, emb_dim, k]
        U_stacked = np.stack(U_per_token)  # [seq_len, output_dim, k]
        S_stacked = np.stack(S_per_token)  # [seq_len, k]
        V_stacked = np.stack(V_per_token)  # [seq_len, emb_dim, k]

        if debug:
            print(f"Final shapes - U: {U_stacked.shape}, S: {S_stacked.shape}, V: {V_stacked.shape}")

            # Optional: print memory usage for debugging
            if debug and torch.cuda.is_available():
                print(f"GPU memory after token {token_idx}: "
                      f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated, "
                      f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB reserved")

                # Clear garbage collector to ensure we see all tensors
                gc.collect()
                min_mb=400
                print(f"\n===== CUDA TENSORS LARGER THAN {min_mb} MB =====")
                
                # Find all tensors
                total_count = 0
                large_tensors = []
                
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.is_cuda:
                            # Calculate size in MB
                            size_mb = obj.element_size() * obj.nelement() / (1024 * 1024)
                            total_count += 1
                            
                            if size_mb >= min_mb:
                                large_tensors.append((size_mb, tuple(obj.shape), obj.dtype))
                    except:
                        # Skip objects that can't be processed
                        pass
                
                # Sort by size (largest first)
                large_tensors.sort(reverse=True)
                
                if large_tensors:
                    print(f"{'SIZE (MB)':<15} {'SHAPE':<30} {'DTYPE':<10}")
                    print("-" * 55)
                    
                    for size_mb, shape, dtype in large_tensors:
                        print(f"{size_mb:<15.2f} {str(shape):<30} {str(dtype):<10}")
                else:
                    print(f"No tensors larger than {min_mb} MB found")
                    
                # Print summary
                print("-" * 55)
                print(f"Total CUDA tensors: {total_count}")
        return U_stacked, S_stacked, V_stacked


def run_comprehensive_tests():
    """Run comprehensive tests on various problem types."""

    print("🧪 COMPREHENSIVE RANDOMIZED SVD TESTS")
    print("="*60)

    # Test 1: Simple linear function
    print("\nTEST 1: Linear transformation")
    print("-" * 30)

    def linear_func(x):
        W = torch.tensor([[2.0, -1.0, 0.5], [1.0, 1.0, -1.0]], dtype=torch.float32)
        return x @ W.T

    x1 = torch.randn(3, 3, dtype=torch.float32, requires_grad=True)

    validate_randomized_svd(linear_func, x1, num_singular_vectors=2, test_ground_truth=True)

    # Test 2: Nonlinear function
    print("\n\nTEST 2: Nonlinear neural network")
    print("-" * 30)

    def nonlinear_func(x):
        W1 = torch.tensor([[1.0, -0.5], [0.5, 1.0], [-1.0, 0.5]], dtype=torch.float32)
        b1 = torch.tensor([0.1, -0.1, 0.2], dtype=torch.float32)
        h = torch.tanh(x @ W1.T + b1)

        W2 = torch.tensor([[1.0, 0.5, -1.0], [0.0, 1.0, 0.5]], dtype=torch.float32)
        b2 = torch.tensor([0.0, 0.1], dtype=torch.float32)
        return h @ W2.T + b2

    x2 = torch.randn(4, 2, dtype=torch.float32, requires_grad=True)

    validate_randomized_svd(nonlinear_func, x2, num_singular_vectors=2, test_ground_truth=True)

    # Test 3: Larger problem (no ground truth)
    print("\n\nTEST 3: Larger problem (randomized only)")
    print("-" * 30)

    # Generate fixed random weights OUTSIDE the function
    torch.manual_seed(42)
    W1_large = torch.randn(30, 50, dtype=torch.float32)
    b1_large = torch.randn(30, dtype=torch.float32)
    W2_large = torch.randn(10, 30, dtype=torch.float32)
    b2_large = torch.randn(10, dtype=torch.float32)

    # Function with pre-defined weights (no randomness)
    def large_func(x):
        h = F.relu(x @ W1_large.T + b1_large)
        return torch.sigmoid(h @ W2_large.T + b2_large)

    x3 = torch.randn(8, 50, dtype=torch.float32, requires_grad=True)

    # Compare original and vectorized implementations directly
    print("Original implementation:")
    start_time = time.time()
    U_orig, S_orig, V_orig = randomized_svd_jacobian_improved(
        large_func, x3, num_singular_vectors=5, debug=False
    )
    orig_time = time.time() - start_time
    print(f"Original implementation time: {orig_time:.4f} seconds")
    print(f"Original SVD singular values: {S_orig.detach().numpy()}")

    print("\nVectorized implementation:")
    start_time = time.time()
    U_vect, S_vect, V_vect = randomized_svd_jacobian_vectorized(
        large_func, x3, num_singular_vectors=5, debug=False
    )
    vect_time = time.time() - start_time
    print(f"Vectorized implementation time: {vect_time:.4f} seconds")
    print(f"Vectorized SVD singular values: {S_vect.detach().numpy()}")

    print(f"\nSpeedup: {orig_time/vect_time:.2f}x faster with vectorization")

    # Test 4: Per-token analysis
    print("\n\nTEST 4: Per-token analysis for sequence models")
    print("-" * 30)

    # Create a small sequence input
    seq_len = 5
    emb_dim = 16

    # Generate all random weights for sequence model outside the function
    torch.manual_seed(42)
    W_proj = torch.randn(emb_dim, 64, dtype=torch.float32)
    W_query = torch.randn(64, 32, dtype=torch.float32)
    W_key = torch.randn(64, 32, dtype=torch.float32)
    W_output = torch.randn(64, 20, dtype=torch.float32)

    # Sequence model with pre-defined weights
    def sequence_model(x):
        # x shape: [batch, seq_len, emb_dim]
        batch, seq_len, emb_dim = x.shape

        # Embedding projection
        hidden = torch.tanh(x @ W_proj)  # [batch, seq_len, 64]

        # Simple self-attention
        query = hidden @ W_query  # [batch, seq_len, 32]
        key = hidden @ W_key      # [batch, seq_len, 32]

        # Attention scores and weighted sum
        scores = torch.bmm(query, key.transpose(1, 2))  # [batch, seq_len, seq_len]
        weights = F.softmax(scores / (32 ** 0.5), dim=-1)  # [batch, seq_len, seq_len]

        # Output projection
        context = torch.bmm(weights, hidden)  # [batch, seq_len, 64]
        output = context @ W_output  # [batch, seq_len, 20]

        return output.reshape(batch, -1)  # Flatten sequence dimension for simplicity

    x4 = torch.randn(1, seq_len, emb_dim, dtype=torch.float32, requires_grad=True)

    def sequence_model(x):
        # Simple sequence model (pretend this is an LLM)
        # x shape: [batch, seq_len, emb_dim]
        batch, seq_len, emb_dim = x.shape

        # Embedding projection
        W_proj = torch.randn(emb_dim, 64, dtype=torch.float32)
        hidden = torch.tanh(x @ W_proj)  # [batch, seq_len, 64]

        # Simple self-attention
        query = hidden @ torch.randn(64, 32, dtype=torch.float32)  # [batch, seq_len, 32]
        key = hidden @ torch.randn(64, 32, dtype=torch.float32)    # [batch, seq_len, 32]

        # Attention scores and weighted sum
        scores = torch.bmm(query, key.transpose(1, 2))  # [batch, seq_len, seq_len]
        weights = F.softmax(scores / (32 ** 0.5), dim=-1)  # [batch, seq_len, seq_len]

        # Output projection
        context = torch.bmm(weights, hidden)  # [batch, seq_len, 64]
        output = context @ torch.randn(64, 20, dtype=torch.float32)  # [batch, seq_len, 20]

        return output.reshape(batch, -1)  # Flatten sequence dimension for simplicity