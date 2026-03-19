import torch


def ensure_size(input, dim, size):
    if input.shape[dim] != size:
        return input.narrow(dim, 0, size)
    return input


def chunk_iter(total_size, chunk_size):
    for start in range(0, total_size, chunk_size):
        if start + chunk_size > total_size:
            yield start, total_size - start
        else:
            yield start, chunk_size


@torch.library.custom_op("torch_nn::linear_cross_entropy_chunking", mutates_args=())
def linear_cross_entropy_chunking(
    input: torch.Tensor,
    linear_weight: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    reduction: str,
    label_smoothing: float,
    batch_chunk_size: int,
    grad_inplace: bool,
    compute_input_grad: bool,
    compute_linear_weight_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = input.device
    dtype = input.dtype
    num_batches, in_features = input.shape
    num_classes, _ = linear_weight.shape

    if target.dtype.is_floating_point:
        raise NotImplementedError(
            "LinearCrossEntropyFunction does not support probability targets"
        )
    else:
        neg_weight = -weight
        neg_weight_target = neg_weight.index_select(0, target)
        if reduction == "mean":
            d = -neg_weight_target.sum()
            neg_weight.div_(d)
            neg_weight_target.div_(d)
        elif reduction == "sum":
            pass
        else:
            raise NotImplementedError(
                f"LinearCrossEntropyFunction does not support {reduction=}"
            )

    if label_smoothing > 0.0:
        raise NotImplementedError(
            "LinearCrossEntropyFunction does not support label smoothing"
        )

    # A chunk buffer used to hold logits, softmax of logits:

    X = torch.empty(
        (batch_chunk_size, num_classes),
        device=device,
        dtype=dtype,
        requires_grad=False,
    )
    if compute_input_grad:
        grad_input = torch.empty_like(input, requires_grad=False)
    else:
        grad_input = torch.empty((0,), dtype=dtype, device=device, requires_grad=False)

    if compute_linear_weight_grad:
        grad_linear_weight = torch.zeros_like(linear_weight, requires_grad=False)
        # A chunk buffer used in grad_linear_weight computation:
        G = torch.empty(
            (num_classes, in_features),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
    else:
        G = torch.empty(())
        grad_linear_weight = torch.empty(
            (0,), dtype=dtype, device=device, requires_grad=False
        )

    if reduction in {"mean", "sum"}:
        output = torch.zeros((), device=device, dtype=dtype, requires_grad=False)
    else:
        raise NotImplementedError(
            f"LinearCrossEntropyFunction does not support {reduction=}"
        )

    # chunking along batches dimension:
    for bchunk_start, bchunk_size in chunk_iter(num_batches, batch_chunk_size):
        x = input.narrow(0, bchunk_start, bchunk_size)
        t = target.narrow(0, bchunk_start, bchunk_size)
        neg_weight_t = neg_weight_target.narrow(0, bchunk_start, bchunk_size)
        X_ = ensure_size(X, 0, bchunk_size)

        # Compute output.
        torch.mm(x, linear_weight.T, out=X_)  # projection

        Xmax = X_.max(dim=1, keepdim=True)[0]

        X_.sub_(Xmax)

        output.add_(neg_weight_t.dot(X_.gather(1, t.unsqueeze(1)).squeeze(1)))

        X_.exp_()

        expXsum = X_.sum(dim=1)

        if compute_input_grad or compute_linear_weight_grad:
            # X_ content will be reused in the classes
            # chunking for-loop below
            X_.mul_((neg_weight_t / expXsum).unsqueeze(1))

        expXsum.log_()

        output.sub_(neg_weight_t.dot(expXsum))

        # Compute gradients.

        if compute_input_grad or compute_linear_weight_grad:
            if compute_input_grad:
                grad_x = grad_input.narrow(0, bchunk_start, bchunk_size)
                torch.index_select(linear_weight, 0, t, out=grad_x)
                grad_x.mul_(neg_weight_t.unsqueeze(1))
            else:
                grad_x = None

            X_ = ensure_size(X_, 1, num_classes)

            if grad_x is not None:
                grad_x.addmm_(X_, linear_weight, alpha=-1)

            if compute_linear_weight_grad:
                grad_L_ = grad_linear_weight.narrow(0, 0, num_classes)
                G.zero_()
                G.index_add_(0, t, x)
                G.mul_(neg_weight.unsqueeze(1))
                G.addmm_(X_.T, x, alpha=-1)
                grad_L_.narrow(1, 0, in_features).add_(G)

    return output, grad_input, grad_linear_weight


@linear_cross_entropy_chunking.register_fake
def _(
    input,
    linear_weight,
    target,
    weight,
    reduction,
    label_smoothing,
    batch_chunk_size,
    grad_inplace,
    compute_input_grad,
    compute_linear_weight_grad,
):
    if reduction in {"mean", "sum"}:
        result = torch.empty((), dtype=input.dtype, device=input.device)
    else:
        raise NotImplementedError(
            f"LinearCrossEntropyFunction does not support {reduction=}"
        )
    if compute_input_grad:
        grad_input = torch.empty_like(input)
    else:
        grad_input = torch.empty(
            (0,), dtype=input.dtype, device=input.device, requires_grad=False
        )
    if compute_linear_weight_grad:
        grad_linear_weight = torch.empty_like(linear_weight)
    else:
        grad_linear_weight = torch.empty(
            (0,),
            dtype=linear_weight.dtype,
            device=linear_weight.device,
            requires_grad=False,
        )
    return result, grad_input, grad_linear_weight


def setup_context(ctx, inputs, output):
    ctx.grad_inplace, ctx.compute_input_grad, ctx.compute_linear_weight_grad = inputs[
        -3:
    ]
    _, grad_input, grad_linear_weight = output
    save_indices: list[int | None] = [None, None]
    saved = []
    if ctx.compute_input_grad:
        save_indices[0] = len(saved)
        saved.append(grad_input)
    if ctx.compute_linear_weight_grad:
        save_indices[1] = len(saved)
        saved.append(grad_linear_weight)
    if saved:
        ctx.save_indices = save_indices
        ctx.save_for_backward(*saved)


def linear_cross_entropy_chunking_backward(ctx, *grads):
    grad_output = grads[0]
    result = [None] * 10

    if ctx.compute_input_grad or ctx.compute_linear_weight_grad:
        saved = ctx.saved_tensors
        if ctx.grad_inplace:
            # With grad_inplace, the memory usage size is reduced
            # 2x when reusing pre-computed grad_input and
            # grad_linear_weight storages. However, gradcheck does
            # not like that.
            if ctx.compute_input_grad:
                grad_input = saved[ctx.save_indices[0]]
                grad_input.mul_(grad_output)
                result[0] = grad_input
            if ctx.compute_linear_weight_grad:
                grad_linear_weight = saved[ctx.save_indices[1]]
                grad_linear_weight.mul_(grad_output)
                result[1] = grad_linear_weight
        else:
            # gradcheck-friendly backward:
            if ctx.compute_input_grad:
                grad_input = saved[ctx.save_indices[0]]
                # creates a new tensor that increases memory usage size
                result[0] = grad_input * grad_output
            if ctx.compute_linear_weight_grad:
                grad_linear_weight = saved[ctx.save_indices[1]]
                # creates a new tensor that increases memory usage size
                result[1] = grad_linear_weight * grad_output

    return tuple(result)


linear_cross_entropy_chunking.register_autograd(
    linear_cross_entropy_chunking_backward, setup_context=setup_context
)
