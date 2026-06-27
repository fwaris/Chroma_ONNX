"""ONNX export and runtime validation helpers."""

from __future__ import annotations

import inspect
import json
from pathlib import Path


class ExportFailure(RuntimeError):
    pass


def export_with_torch(
    torch,
    module,
    args: tuple,
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
    external_data: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "opset_version": opset,
        "do_constant_folding": True,
    }
    export_parameters = inspect.signature(torch.onnx.export).parameters
    if "external_data" in export_parameters:
        export_kwargs["external_data"] = external_data
    elif "use_external_data_format" in export_parameters:
        export_kwargs["use_external_data_format"] = external_data

    try:
        with torch.no_grad():
            torch.onnx.export(
                module,
                args,
                str(output_path),
                **export_kwargs,
            )
    except Exception as exc:
        raise ExportFailure(f"Failed to export {output_path.name}: {exc}") from exc


def check_onnx(path: Path) -> None:
    import onnx

    onnx.checker.check_model(str(path))


def fix_onnxruntime_compat(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(path), load_external_data=False)
    producers = {output: node for node in model.graph.node for output in node.output}
    fixed = False
    new_nodes = []

    for node in model.graph.node:
        if node.op_type == "CumSum":
            producer = producers.get(node.input[0])
            if producer is not None and producer.op_type == "Not":
                cast_output = f"{node.input[0]}_int64_for_cumsum"
                cast_node = helper.make_node(
                    "Cast",
                    [node.input[0]],
                    [cast_output],
                    name=f"{node.name or 'CumSum'}_CastBoolInputToInt64",
                    to=TensorProto.INT64,
                )
                new_nodes.append(cast_node)
                node.input[0] = cast_output
                fixed = True
        new_nodes.append(node)

    if fixed:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        onnx.save_model(model, str(path))


def validate_onnxruntime(path: Path, inputs: dict[str, object]) -> list[str]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, inputs)
    return [str(output.shape) for output in outputs]


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def module_dtype(module, default):
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return default


def configure_quality_safe_float32(torch) -> None:
    """Use deterministic full-FP32 math for Python reference/export parity with ORT."""

    if hasattr(torch.backends, "cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
    if hasattr(torch.backends, "cudnn"):
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass


def install_onnx_safe_torch_ops(torch) -> None:
    """Lower a few ops to forms that the legacy ONNX exporter handles better."""

    configure_quality_safe_float32(torch)

    original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention

    def scaled_dot_product_attention(*args, **kwargs):
        args = list(args)
        if len(args) >= 4:
            attn_mask = args[3]
            if attn_mask is not None and hasattr(attn_mask, "is_contiguous") and not attn_mask.is_contiguous():
                args[3] = attn_mask.contiguous()
        elif "attn_mask" in kwargs:
            attn_mask = kwargs["attn_mask"]
            if attn_mask is not None and hasattr(attn_mask, "is_contiguous") and not attn_mask.is_contiguous():
                kwargs["attn_mask"] = attn_mask.contiguous()
        return original_scaled_dot_product_attention(*args, **kwargs)

    def diff(input, n=1, dim=-1, prepend=None, append=None):
        if prepend is not None:
            input = torch.cat((prepend, input), dim=dim)
        if append is not None:
            input = torch.cat((input, append), dim=dim)

        result = input
        rank = result.dim()
        resolved_dim = dim if dim >= 0 else rank + dim
        for _ in range(n):
            size = result.shape[resolved_dim]
            upper = result.narrow(resolved_dim, 1, size - 1)
            lower = result.narrow(resolved_dim, 0, size - 1)
            result = upper - lower
        return result

    def cdist(x1, x2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"):
        if float(p) != 2.0:
            raise NotImplementedError("The ONNX-safe torch.cdist shim only supports p=2.")

        x1 = x1.float()
        x2 = x2.float()
        # Mimi codec prompt encoding is RVQ/argmin based, so tiny matmul-order
        # differences can flip a boundary code ID. The direct formulation is
        # heavier but tracks PyTorch cdist code selection more closely.
        delta = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        distances_squared = (delta * delta).sum(dim=-1)
        return torch.sqrt(torch.clamp(distances_squared, min=0.0))

    torch.diff = diff
    torch.cdist = cdist
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention


def write_manifest(output_dir: Path, manifest: dict) -> None:
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
