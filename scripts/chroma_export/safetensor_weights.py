"""Safetensors-backed ONNX weight sharing."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from copy import deepcopy
from pathlib import Path

from .dependencies import require_safetensors
from .onnx_export import ExportFailure


SHARED_GRAPH_FILES = {
    "system_prefill": "chroma_system_prefill",
    "decoder": "chroma_decoder",
    "codec_decode": "chroma_codec_decode",
}

S2S_REQUIRED_GRAPH_FILES = {
    "generate_prefill": "chroma_generate_prefill",
    "backbone_frame_step": "chroma_backbone_frame_step",
    "backbone_thinker_step": "chroma_backbone_thinker_step",
    "decoder": "chroma_decoder",
    "decoder_prefill": "chroma_decoder_prefill",
    "decoder_step": "chroma_decoder_step",
    "codec_decode": "chroma_codec_decode",
}

S2S_MERGED_GRAPH_NAME = "s2s_merged"
S2S_MERGED_GRAPH_FILE = "chroma_s2s_merged"


def tensor_num_bytes(tensor) -> int:
    import numpy as np
    import onnx

    element_sizes = {
        onnx.TensorProto.FLOAT: 4,
        onnx.TensorProto.UINT8: 1,
        onnx.TensorProto.INT8: 1,
        onnx.TensorProto.UINT16: 2,
        onnx.TensorProto.INT16: 2,
        onnx.TensorProto.INT32: 4,
        onnx.TensorProto.INT64: 8,
        onnx.TensorProto.BOOL: 1,
        onnx.TensorProto.FLOAT16: 2,
        onnx.TensorProto.DOUBLE: 8,
        onnx.TensorProto.UINT32: 4,
        onnx.TensorProto.UINT64: 8,
        onnx.TensorProto.BFLOAT16: 2,
    }
    return int(np.prod(tensor.dims, dtype=np.int64)) * element_sizes.get(tensor.data_type, 0)


def direct_safetensor_candidates(onnx_name: str) -> list[str]:
    candidates = [onnx_name]
    if onnx_name.startswith("chroma."):
        candidates.append(onnx_name[len("chroma.") :])
    return candidates


class SafetensorIndex:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.safe_open = require_safetensors()
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise SystemExit(f"Missing safetensors index: {index_path}")
        self.weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, str] = {}
        self.offsets: dict[str, int] = {}
        self.byte_lengths: dict[str, int] = {}
        self.by_shape: dict[tuple[int, ...], list[str]] = {}
        self._hash_cache: dict[tuple[str, str], str] = {}

        for shard in sorted(set(self.weight_map.values())):
            shard_path = model_dir / shard
            with shard_path.open("rb") as stream:
                header_length = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(header_length))
                data_start = 8 + header_length

            for name, metadata in header.items():
                if name == "__metadata__":
                    continue

                shape = tuple(int(dim) for dim in metadata["shape"])
                dtype = str(metadata["dtype"])
                begin, end = (int(value) for value in metadata["data_offsets"])
                self.shapes[name] = shape
                self.dtypes[name] = dtype
                self.offsets[name] = data_start + begin
                self.byte_lengths[name] = end - begin
                self.by_shape.setdefault(shape, []).append(name)

    def shard_for(self, source_tensor: str) -> str:
        return self.weight_map[source_tensor]

    def array(self, source_tensor: str):
        with self.safe_open(self.model_dir / self.shard_for(source_tensor), framework="np") as handle:
            return handle.get_tensor(source_tensor)

    def hash_for(self, source_tensor: str, transform: str) -> str:
        import numpy as np

        key = (source_tensor, transform)
        if key not in self._hash_cache:
            array = self.array(source_tensor)
            if transform == "transpose":
                array = array.T
            if self.dtypes[source_tensor] == "BF16":
                array = array.astype(np.float32)
            self._hash_cache[key] = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        return self._hash_cache[key]


def find_safetensor_mapping(tensor, safetensors: SafetensorIndex):
    import onnx
    from onnx import numpy_helper

    if tensor.data_type != onnx.TensorProto.FLOAT:
        return None

    onnx_shape = tuple(int(dim) for dim in tensor.dims)
    for candidate in direct_safetensor_candidates(tensor.name):
        if (
            candidate in safetensors.shapes
            and safetensors.shapes[candidate] == onnx_shape
            and safetensors.dtypes[candidate] in {"F32", "BF16"}
        ):
            return candidate, "identity"

    array = numpy_helper.to_array(tensor)
    onnx_hash = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    candidates: list[tuple[str, str]] = []
    candidates.extend(
        (name, "identity")
        for name in safetensors.by_shape.get(onnx_shape, [])
        if safetensors.dtypes[name] in {"F32", "BF16"}
    )
    if len(onnx_shape) == 2:
        transposed_shape = (onnx_shape[1], onnx_shape[0])
        candidates.extend(
            (name, "transpose")
            for name in safetensors.by_shape.get(transposed_shape, [])
            if safetensors.dtypes[name] in {"F32", "BF16"}
        )

    for source_tensor, transform in candidates:
        if safetensors.hash_for(source_tensor, transform) == onnx_hash:
            return source_tensor, transform
    return None


def onnx_dtype_for_safetensor(dtype: str) -> int:
    import onnx

    if dtype == "F32":
        return onnx.TensorProto.FLOAT
    if dtype == "BF16":
        return onnx.TensorProto.BFLOAT16
    raise ExportFailure(f"Unsupported safetensors dtype for shared initializer: {dtype}")


def ensure_hardlinked_safetensor(source_path: Path, output_dir: Path) -> Path:
    target_path = output_dir / source_path.name
    if source_path.resolve() == target_path.resolve():
        return target_path

    if target_path.exists():
        try:
            if os.path.samefile(source_path, target_path):
                return target_path
        except OSError:
            pass
        raise ExportFailure(
            f"Cannot create safetensors hardlink because {target_path} already exists "
            f"and is not the same file as {source_path}."
        )

    try:
        os.link(source_path, target_path)
    except OSError as exc:
        raise ExportFailure(
            f"Failed to hardlink {source_path} into {output_dir}. "
            "Place the shared ONNX bundle on the same volume as the safetensors, "
            "or create the hardlinks manually."
        ) from exc
    return target_path


def external_data_location(source_path: Path, output_path: Path) -> str:
    try:
        return Path(os.path.relpath(source_path, output_path.parent)).as_posix()
    except ValueError:
        return str(source_path)


def point_tensor_to_safetensor_data(
    tensor,
    source_shape: tuple[int, ...],
    source_dtype: str,
    source_shard_path: Path,
    output_path: Path,
    source_offset: int,
    byte_length: int,
) -> None:
    import onnx

    del tensor.dims[:]
    tensor.dims.extend(source_shape)
    tensor.data_type = onnx_dtype_for_safetensor(source_dtype)
    tensor.ClearField("raw_data")
    tensor.ClearField("float_data")
    tensor.ClearField("int32_data")
    tensor.ClearField("int64_data")
    tensor.ClearField("double_data")
    tensor.ClearField("uint64_data")
    tensor.ClearField("string_data")
    del tensor.external_data[:]
    tensor.data_location = onnx.TensorProto.EXTERNAL
    # ONNX Runtime validates external_data locations before supplied shared
    # initializers are registered. Keep locations inside the graph directory;
    # the F# runtime creates ignored local hardlinks to the configured model dir.
    location = source_shard_path.name
    for key, value in (
        ("location", location),
        ("offset", str(source_offset)),
        ("length", str(byte_length)),
    ):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value


def replace_initializer_uses(model, initializer_name: str, replacement_name: str) -> None:
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == initializer_name:
                node.input[index] = replacement_name


def replace_value_uses(model, value_name: str, replacement_name: str, skip_node=None) -> None:
    for node in model.graph.node:
        if skip_node is not None and node is skip_node:
            continue
        for index, input_name in enumerate(node.input):
            if input_name == value_name:
                node.input[index] = replacement_name


def rewrite_bfloat16_gather_uses(model, initializer_name: str) -> bool:
    from onnx import TensorProto, helper

    consumers = [node for node in model.graph.node if initializer_name in node.input]
    if not consumers or any(node.op_type != "Gather" for node in consumers):
        return False

    inserts: list[tuple[int, object]] = []
    for node in consumers:
        node_index = list(model.graph.node).index(node)
        for output_name in node.output:
            cast_output = f"{output_name}_float"
            replace_value_uses(model, output_name, cast_output, skip_node=node)
            inserts.append(
                (
                    node_index + 1,
                    helper.make_node(
                        "Cast",
                        [output_name],
                        [cast_output],
                        name=f"{output_name}_CastBFloat16ToFloat",
                        to=TensorProto.FLOAT,
                    ),
                )
            )

    for index, node in sorted(inserts, key=lambda item: item[0], reverse=True):
        model.graph.node.insert(index, node)
    return True


def rewrite_initializer_uses(model, initializer_name: str, transform: str, source_dtype: str) -> None:
    from onnx import TensorProto, helper

    if transform == "identity" and source_dtype == "F32":
        return

    if transform == "identity" and source_dtype == "BF16" and rewrite_bfloat16_gather_uses(model, initializer_name):
        return

    current_name = initializer_name
    final_name = initializer_name
    nodes = []

    if source_dtype == "BF16":
        cast_name = f"{initializer_name}_float"
        nodes.append(
            helper.make_node(
                "Cast",
                [current_name],
                [cast_name],
                name=f"{initializer_name}_CastBFloat16ToFloat",
                to=TensorProto.FLOAT,
            )
        )
        current_name = cast_name
        final_name = cast_name

    if transform == "transpose":
        transposed_name = f"{initializer_name}_transposed"
        nodes.append(
            helper.make_node(
                "Transpose",
                [current_name],
                [transposed_name],
                name=f"{initializer_name}_TransposeFromSafetensors",
                perm=[1, 0],
            )
        )
        final_name = transposed_name
    elif transform != "identity":
        raise ExportFailure(f"Unsupported initializer transform: {transform}")

    replace_initializer_uses(model, initializer_name, final_name)
    for node in reversed(nodes):
        model.graph.node.insert(0, node)


def convert_graph_to_safetensor_shared(
    source_path: Path,
    output_path: Path,
    graph_name: str,
    safetensors: SafetensorIndex,
    large_threshold_bytes: int = 1024 * 1024,
) -> list[dict[str, object]]:
    import onnx

    model = onnx.load(str(source_path), load_external_data=True)
    entries: list[dict[str, object]] = []
    seen: set[str] = set()

    for tensor in model.graph.initializer:
        mapping = find_safetensor_mapping(tensor, safetensors)
        size_bytes = tensor_num_bytes(tensor)
        if mapping is None:
            if size_bytes > large_threshold_bytes:
                raise ExportFailure(
                    f"Could not map large initializer {tensor.name!r} in {source_path.name} "
                    f"with shape {list(tensor.dims)} and {size_bytes} bytes to safetensors."
                )
            continue

        source_tensor, transform = mapping
        source_shape = safetensors.shapes[source_tensor]
        source_dtype = safetensors.dtypes[source_tensor]
        source_shard = safetensors.shard_for(source_tensor)
        source_offset = safetensors.offsets[source_tensor]
        source_byte_length = safetensors.byte_lengths[source_tensor]
        rewrite_initializer_uses(model, tensor.name, transform, source_dtype)
        point_tensor_to_safetensor_data(
            tensor,
            source_shape,
            source_dtype,
            safetensors.model_dir / source_shard,
            output_path,
            source_offset,
            source_byte_length,
        )
        expected_byte_length = tensor_num_bytes(tensor)
        if expected_byte_length != source_byte_length:
            raise ExportFailure(
                f"Safetensors byte length mismatch for {source_tensor}: "
                f"ONNX metadata expects {expected_byte_length}, safetensors span is {source_byte_length}."
            )

        if tensor.name not in seen:
            seen.add(tensor.name)
            entries.append(
                {
                    "graph": graph_name,
                    "onnx_initializer": tensor.name,
                    "source_shard": source_shard,
                    "source_tensor": source_tensor,
                    "dtype": source_dtype,
                    "shape": list(source_shape),
                    "source_offset": source_offset,
                    "byte_length": source_byte_length,
                    "transform": transform,
                }
            )

    fix_onnxruntime_compatibility_for_model(model)
    onnx.save_model(model, str(output_path), save_as_external_data=False)
    return entries


def fix_onnxruntime_compatibility_for_model(model) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    producers = {output: node for node in model.graph.node for output in node.output}
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    new_nodes = []

    def tensor_attr(node, name: str):
        for attr in node.attribute:
            if attr.name == name and attr.type == onnx.AttributeProto.TENSOR:
                return attr.t
        return None

    def set_tensor_attr(node, name: str, tensor) -> bool:
        for attr in node.attribute:
            if attr.name == name and attr.type == onnx.AttributeProto.TENSOR:
                attr.t.CopyFrom(tensor)
                return True
        return False

    def value_dtype(value_name: str):
        initializer = initializers.get(value_name)
        if initializer is not None:
            return initializer.data_type
        producer = producers.get(value_name)
        if producer is not None and producer.op_type == "Constant":
            tensor = tensor_attr(producer, "value")
            if tensor is not None:
                return tensor.data_type
        return None

    def convert_constant_or_initializer_to_int64(value_name: str) -> bool:
        initializer = initializers.get(value_name)
        if initializer is not None and initializer.data_type != TensorProto.INT64:
            array = numpy_helper.to_array(initializer).astype(np.int64)
            initializer.CopyFrom(numpy_helper.from_array(array, initializer.name))
            return True

        producer = producers.get(value_name)
        if producer is not None and producer.op_type == "Constant":
            tensor = tensor_attr(producer, "value")
            if tensor is not None and tensor.data_type != TensorProto.INT64:
                array = numpy_helper.to_array(tensor).astype(np.int64)
                set_tensor_attr(producer, "value", numpy_helper.from_array(array))
                return True

        return False

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
        if (
            node.op_type == "AveragePool"
            and "audio_tower/avg_pooler" in node.name
            and node.input
            and not node.input[0].endswith("_unsqueezed_for_avgpool")
        ):
            original_input = node.input[0]
            original_output = node.output[0]
            axes_name = f"{node.name}_Axes0".replace("/", "_")
            unsqueezed_input = f"{original_input}_unsqueezed_for_avgpool"
            unsqueezed_output = f"{original_output}_unsqueezed_for_avgpool"
            axes_node = helper.make_node(
                "Constant",
                [],
                [axes_name],
                name=f"{node.name}_Axes0Constant",
                value=numpy_helper.from_array(np.array([0], dtype=np.int64)),
            )
            unsqueeze_node = helper.make_node(
                "Unsqueeze",
                [original_input, axes_name],
                [unsqueezed_input],
                name=f"{node.name}_UnsqueezeBatchForOnnxRuntime",
            )
            node.input[0] = unsqueezed_input
            node.output[0] = unsqueezed_output
            squeeze_node = helper.make_node(
                "Squeeze",
                [unsqueezed_output, axes_name],
                [original_output],
                name=f"{node.name}_SqueezeBatchForOnnxRuntime",
            )
            new_nodes.append(axes_node)
            new_nodes.append(unsqueeze_node)
            new_nodes.append(node)
            new_nodes.append(squeeze_node)
            continue
        if node.op_type == "Slice":
            for input_index in range(1, len(node.input)):
                value_name = node.input[input_index]
                if not value_name:
                    continue
                dtype = value_dtype(value_name)
                if dtype == TensorProto.INT64:
                    continue
                if dtype is not None and convert_constant_or_initializer_to_int64(value_name):
                    continue

                cast_output = f"{value_name}_int64_for_slice"
                cast_node = helper.make_node(
                    "Cast",
                    [value_name],
                    [cast_output],
                    name=f"{node.name or 'Slice'}_CastInput{input_index}ToInt64",
                    to=TensorProto.INT64,
                )
                new_nodes.append(cast_node)
                node.input[input_index] = cast_output
        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    seen_node_names: dict[str, int] = {}
    for node in model.graph.node:
        if not node.name:
            continue
        count = seen_node_names.get(node.name, 0)
        seen_node_names[node.name] = count + 1
        if count > 0:
            node.name = f"{node.name}_dedup_{count}"


def _external_initializer_signature(tensor) -> tuple[object, ...]:
    return (
        tensor.data_type,
        tuple(tensor.dims),
        tensor.data_location,
        tuple(sorted((entry.key, entry.value) for entry in tensor.external_data)),
    )


def _external_initializer_length(tensor) -> int:
    for entry in tensor.external_data:
        if entry.key == "length":
            return int(entry.value)
    return tensor_num_bytes(tensor)


def _is_readable_initializer_name(name: str) -> bool:
    return bool(name) and not name.startswith("onnx::")


def _prefix_s2s_value(
    graph_name: str,
    value_name: str,
    initializer_names: set[str],
    initializer_name_map: dict[str, str] | None = None,
) -> str:
    if not value_name:
        return value_name
    if value_name in initializer_names:
        if initializer_name_map is not None:
            return initializer_name_map.get(value_name, value_name)
        return value_name
    return f"{graph_name}__{value_name}"


def _prefix_nested_graph_values(graph, graph_name: str, initializer_names: set[str], initializer_name_map: dict[str, str]) -> None:
    import onnx

    local_initializers = initializer_names | {initializer.name for initializer in graph.initializer}
    for value in graph.input:
        if value.name not in local_initializers:
            value.name = _prefix_s2s_value(graph_name, value.name, local_initializers, initializer_name_map)
    for value in graph.output:
        if value.name not in local_initializers:
            value.name = _prefix_s2s_value(graph_name, value.name, local_initializers, initializer_name_map)
    for value in graph.value_info:
        if value.name not in local_initializers:
            value.name = _prefix_s2s_value(graph_name, value.name, local_initializers, initializer_name_map)

    for node_index, node in enumerate(graph.node):
        node.name = f"{graph_name}__{node.name or node.op_type}_{node_index}"
        node.input[:] = [
            _prefix_s2s_value(graph_name, value_name, local_initializers, initializer_name_map)
            for value_name in node.input
        ]
        node.output[:] = [
            _prefix_s2s_value(graph_name, value_name, local_initializers, initializer_name_map)
            for value_name in node.output
        ]
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                _prefix_nested_graph_values(attr.g, graph_name, initializer_names, initializer_name_map)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for nested in attr.graphs:
                    _prefix_nested_graph_values(nested, graph_name, initializer_names, initializer_name_map)


def create_resident_merged_s2s_model(
    output_dir: Path,
    shared_graphs: dict[str, dict[str, object]],
    output_path: Path | None = None,
) -> dict[str, object]:
    """Merge S2S component graphs into one resident ORT graph.

    The subgraphs stay disconnected, with graph-specific prefixes on public and
    internal values. Initializers keep their original names so the merged graph
    has one initializer namespace and one ORT session can own the weights.
    """

    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    missing = [name for name in S2S_REQUIRED_GRAPH_FILES if name not in shared_graphs]
    if missing:
        raise ExportFailure(f"Cannot create resident merged S2S model. Missing graphs: {', '.join(missing)}")

    output_path = output_path or (output_dir / f"{S2S_MERGED_GRAPH_FILE}.weights_free.onnx")
    merged_inputs = []
    merged_outputs = []
    merged_initializers = {}
    merged_initializer_signatures = {}
    opset_imports = None
    ir_version = None
    graph_inputs: list[str] = []
    graph_outputs: list[str] = []
    initializer_aliases: list[dict[str, str]] = []
    loaded_models = []
    component_nodes: dict[str, list[object]] = {}
    component_value_info: dict[str, list[object]] = {}
    component_outputs: dict[str, dict[str, object]] = {}
    initializer_occurrences: list[dict[str, object]] = []
    occurrences_by_signature: dict[tuple[object, ...], list[dict[str, object]]] = {}

    for graph_name in S2S_REQUIRED_GRAPH_FILES:
        graph_path = output_dir / str(shared_graphs[graph_name]["path"])
        if not graph_path.is_file():
            graph_path = Path(str(shared_graphs[graph_name]["path"]))
        model = onnx.load(str(graph_path), load_external_data=False)
        if opset_imports is None:
            opset_imports = deepcopy(model.opset_import)
            ir_version = model.ir_version
        loaded_models.append((graph_name, model))

    signatures_by_name: dict[str, set[tuple[object, ...]]] = {}
    referenced_initializer_bytes = 0
    for graph_name, model in loaded_models:
        for initializer in model.graph.initializer:
            signature = _external_initializer_signature(initializer)
            byte_length = _external_initializer_length(initializer)
            occurrence = {
                "graph": graph_name,
                "onnx_initializer": initializer.name,
                "initializer": initializer,
                "signature": signature,
                "byte_length": byte_length,
            }
            initializer_occurrences.append(occurrence)
            occurrences_by_signature.setdefault(signature, []).append(occurrence)
            signatures_by_name.setdefault(initializer.name, set()).add(signature)
            referenced_initializer_bytes += byte_length

    conflicting_initializer_names = {
        name for name, signatures in signatures_by_name.items() if len(signatures) > 1
    }

    used_initializer_names: dict[str, tuple[object, ...]] = {}
    canonical_name_by_signature: dict[tuple[object, ...], str] = {}

    def allocate_initializer_name(signature: tuple[object, ...], occurrences: list[dict[str, object]]) -> str:
        seen_candidates = set()
        candidate_names: list[str] = []
        for occurrence in occurrences:
            name = str(occurrence["onnx_initializer"])
            if name not in seen_candidates:
                seen_candidates.add(name)
                candidate_names.append(name)

        preferred_names = (
            [
                name
                for name in candidate_names
                if name not in conflicting_initializer_names and _is_readable_initializer_name(name)
            ]
            + [
                name
                for name in candidate_names
                if name not in conflicting_initializer_names and not _is_readable_initializer_name(name)
            ]
        )

        for name in preferred_names:
            if name not in used_initializer_names:
                used_initializer_names[name] = signature
                return name
            if used_initializer_names[name] == signature:
                return name

        first = occurrences[0]
        fallback_base = f"{first['graph']}__{first['onnx_initializer']}"
        fallback = fallback_base
        suffix = 1
        while fallback in used_initializer_names and used_initializer_names[fallback] != signature:
            suffix += 1
            fallback = f"{fallback_base}__dedup_{suffix}"
        used_initializer_names[fallback] = signature
        return fallback

    for signature, occurrences in occurrences_by_signature.items():
        merged_name = allocate_initializer_name(signature, occurrences)
        canonical_name_by_signature[signature] = merged_name
        merged_initializer = deepcopy(occurrences[0]["initializer"])
        merged_initializer.name = merged_name
        merged_initializers[merged_name] = merged_initializer
        merged_initializer_signatures[merged_name] = signature

    duplicate_groups = [
        (signature, occurrences)
        for signature, occurrences in occurrences_by_signature.items()
        if len(occurrences) > 1
    ]
    unique_initializer_bytes = sum(_external_initializer_length(tensor) for tensor in merged_initializers.values())
    duplicate_initializer_bytes_removed = referenced_initializer_bytes - unique_initializer_bytes
    top_duplicate_groups = []
    for signature, occurrences in sorted(
        duplicate_groups,
        key=lambda item: (
            _external_initializer_length(item[1][0]["initializer"]) * (len(item[1]) - 1)
        ),
        reverse=True,
    )[:20]:
        byte_length = _external_initializer_length(occurrences[0]["initializer"])
        top_duplicate_groups.append(
            {
                "canonical_initializer": canonical_name_by_signature[signature],
                "byte_length": byte_length,
                "reference_count": len(occurrences),
                "duplicate_bytes_removed": byte_length * (len(occurrences) - 1),
                "aliases": [
                    {
                        "graph": str(occurrence["graph"]),
                        "onnx_initializer": str(occurrence["onnx_initializer"]),
                    }
                    for occurrence in occurrences[:8]
                ],
            }
        )

    merged_signature_counts: dict[tuple[object, ...], int] = {}
    for signature in merged_initializer_signatures.values():
        merged_signature_counts[signature] = merged_signature_counts.get(signature, 0) + 1
    remaining_duplicate_signatures = {
        signature: count for signature, count in merged_signature_counts.items() if count > 1
    }
    if remaining_duplicate_signatures:
        raise ExportFailure(
            "Merged S2S graph still has duplicate external initializers after canonicalization: "
            f"{len(remaining_duplicate_signatures)} duplicate signatures."
        )

    canonicalization_report = {
        "referenced_initializer_count_before": len(initializer_occurrences),
        "initializer_count_after": len(merged_initializers),
        "duplicate_initializer_count_removed": len(initializer_occurrences) - len(merged_initializers),
        "referenced_bytes_before": referenced_initializer_bytes,
        "unique_bytes_after": unique_initializer_bytes,
        "duplicate_bytes_removed": duplicate_initializer_bytes_removed,
        "duplicate_group_count": len(duplicate_groups),
        "conflicting_name_count": len(conflicting_initializer_names),
        "top_duplicate_groups": top_duplicate_groups,
    }
    print(
        "Canonicalized merged S2S initializers: "
        f"{len(initializer_occurrences)} refs -> {len(merged_initializers)} unique, "
        f"removed {duplicate_initializer_bytes_removed / 1024 / 1024 / 1024:.3f} GiB duplicate references."
    )

    for graph_name, model in loaded_models:
        initializer_names = {initializer.name for initializer in model.graph.initializer}
        initializer_name_map = {}

        for initializer in model.graph.initializer:
            signature = _external_initializer_signature(initializer)
            merged_name = canonical_name_by_signature[signature]
            initializer_name_map[initializer.name] = merged_name
            initializer_aliases.append(
                {
                    "graph": graph_name,
                    "onnx_initializer": initializer.name,
                    "merged_initializer": merged_name,
                }
            )

        for value in model.graph.input:
            if value.name in initializer_names:
                continue
            prefixed = deepcopy(value)
            prefixed.name = _prefix_s2s_value(graph_name, value.name, initializer_names, initializer_name_map)
            merged_inputs.append(prefixed)
            graph_inputs.append(prefixed.name)

        component_outputs[graph_name] = {}
        for value in model.graph.output:
            prefixed = deepcopy(value)
            prefixed.name = _prefix_s2s_value(graph_name, value.name, initializer_names, initializer_name_map)
            merged_outputs.append(prefixed)
            graph_outputs.append(prefixed.name)
            component_outputs[graph_name][prefixed.name] = prefixed

        graph_value_info = []
        for value in model.graph.value_info:
            if value.name in initializer_names:
                continue
            prefixed = deepcopy(value)
            prefixed.name = _prefix_s2s_value(graph_name, value.name, initializer_names, initializer_name_map)
            graph_value_info.append(prefixed)
        component_value_info[graph_name] = graph_value_info

        graph_nodes = []
        for node_index, node in enumerate(model.graph.node):
            prefixed_node = deepcopy(node)
            prefixed_node.name = f"{graph_name}__{node.name or node.op_type}_{node_index}"
            prefixed_node.input[:] = [
                _prefix_s2s_value(graph_name, value_name, initializer_names, initializer_name_map)
                for value_name in node.input
            ]
            prefixed_node.output[:] = [
                _prefix_s2s_value(graph_name, value_name, initializer_names, initializer_name_map)
                for value_name in node.output
            ]
            for attr in prefixed_node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    _prefix_nested_graph_values(attr.g, graph_name, initializer_names, initializer_name_map)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for nested in attr.graphs:
                        _prefix_nested_graph_values(nested, graph_name, initializer_names, initializer_name_map)
            graph_nodes.append(prefixed_node)
        component_nodes[graph_name] = graph_nodes

    mode_input_name = "s2s_mode"
    merged_inputs.insert(0, helper.make_tensor_value_info(mode_input_name, TensorProto.INT64, []))

    def dummy_tensor_for_output(value_info, name: str):
        tensor_type = value_info.type.tensor_type
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(1)
        if tensor_type.elem_type == TensorProto.FLOAT:
            array = np.zeros(shape, dtype=np.float32)
        elif tensor_type.elem_type == TensorProto.INT64:
            array = np.zeros(shape, dtype=np.int64)
        elif tensor_type.elem_type == TensorProto.BOOL:
            array = np.zeros(shape, dtype=np.bool_)
        else:
            raise ExportFailure(f"Unsupported merged S2S branch dummy output element type {tensor_type.elem_type}.")
        return numpy_helper.from_array(array, name=name)

    def make_branch_graph(graph_name: str):
        nodes = [deepcopy(node) for node in component_nodes[graph_name]]
        outputs = []
        branch_value_info = [deepcopy(value) for value in component_value_info[graph_name]]
        actual_outputs = component_outputs[graph_name]
        for output_index, output in enumerate(merged_outputs):
            if output.name in actual_outputs:
                outputs.append(deepcopy(actual_outputs[output.name]))
            else:
                dummy_name = f"{graph_name}__dummy_output_{output_index}"
                nodes.append(
                    helper.make_node(
                        "Constant",
                        [],
                        [dummy_name],
                        name=f"{graph_name}__dummy_constant_{output_index}",
                        value=dummy_tensor_for_output(output, f"{dummy_name}_value"),
                    )
                )
                dummy_output = deepcopy(output)
                dummy_output.name = dummy_name
                outputs.append(dummy_output)
        return helper.make_graph(
            nodes,
            f"{graph_name}_branch",
            [],
            outputs,
            value_info=branch_value_info,
        )

    graph_order = list(S2S_REQUIRED_GRAPH_FILES.keys())

    def make_selector_graph(index: int):
        graph_name = graph_order[index]
        if index == len(graph_order) - 1:
            return make_branch_graph(graph_name)

        condition_const = f"selector_{index}__mode_value"
        condition_output = f"selector_{index}__condition"
        if_outputs = [f"selector_{index}__{output.name}" for output in merged_outputs]
        nodes = [
            helper.make_node(
                "Constant",
                [],
                [condition_const],
                name=f"selector_{index}__mode_constant",
                value=numpy_helper.from_array(np.array(index, dtype=np.int64), name=f"selector_{index}__mode_value_tensor"),
            ),
            helper.make_node(
                "Equal",
                [mode_input_name, condition_const],
                [condition_output],
                name=f"selector_{index}__mode_equal",
            ),
            helper.make_node(
                "If",
                [condition_output],
                if_outputs,
                name=f"selector_{index}__dispatch",
                then_branch=make_branch_graph(graph_name),
                else_branch=make_selector_graph(index + 1),
            ),
        ]
        outputs = []
        for name, output in zip(if_outputs, merged_outputs):
            value = deepcopy(output)
            value.name = name
            outputs.append(value)
        return helper.make_graph(nodes, f"selector_{index}_branch", [], outputs)

    top_condition_const = "s2s_mode_0_value"
    top_condition_output = "s2s_mode_is_0"
    merged_nodes = [
        helper.make_node(
            "Constant",
            [],
            [top_condition_const],
            name="s2s_mode_0_constant",
            value=numpy_helper.from_array(np.array(0, dtype=np.int64), name="s2s_mode_0_value_tensor"),
        ),
        helper.make_node(
            "Equal",
            [mode_input_name, top_condition_const],
            [top_condition_output],
            name="s2s_mode_0_equal",
        ),
        helper.make_node(
            "If",
            [top_condition_output],
            [output.name for output in merged_outputs],
            name="s2s_dispatch",
            then_branch=make_branch_graph(graph_order[0]),
            else_branch=make_selector_graph(1),
        ),
    ]

    graph = helper.make_graph(
        merged_nodes,
        "chroma_s2s_resident_merged",
        merged_inputs,
        merged_outputs,
        initializer=list(merged_initializers.values()),
    )
    model = helper.make_model(
        graph,
        producer_name="chroma-s2s-resident-merge",
        opset_imports=opset_imports,
    )
    if ir_version is not None:
        model.ir_version = ir_version
    # The no-copy shared bundle intentionally points external_data locations
    # back to the original model directory. ONNX checker rejects paths outside
    # the model directory even though ONNX Runtime can load them correctly.
    onnx.save_model(model, str(output_path), save_as_external_data=False)

    return {
        "path": str(output_path),
        "inputs": graph_inputs,
        "outputs": graph_outputs,
        "description": "Resident merged S2S graph with prefixed disconnected component subgraphs.",
        "initializer_count": len(merged_initializers),
        "component_graphs": list(S2S_REQUIRED_GRAPH_FILES.keys()),
        "input_prefix": "{graph}__",
        "output_prefix": "{graph}__",
        "initializer_aliases": initializer_aliases,
        "initializer_canonicalization": canonicalization_report,
    }


def create_shared_bundle(
    output_dir: Path,
    model_dir: Path,
    manifest: dict,
    *,
    bundle_name: str = "safetensor-shared-e2e",
    graph_files: dict[str, str] | None = None,
    capabilities: dict[str, object] | None = None,
    single_model_only: bool = False,
) -> dict[str, object]:
    safetensors = SafetensorIndex(model_dir)
    shared_graphs: dict[str, dict[str, object]] = {}
    initializers: list[dict[str, object]] = []
    graph_files = graph_files or SHARED_GRAPH_FILES

    for graph_name, base_name in graph_files.items():
        if graph_name not in manifest["graphs"]:
            raise ExportFailure(f"Shared bundle requires graph {graph_name!r}, but it was not exported.")
        source_path = output_dir / f"{base_name}.onnx"
        output_path = output_dir / f"{base_name}.weights_free.onnx"
        graph_entries = convert_graph_to_safetensor_shared(source_path, output_path, graph_name, safetensors)
        graph_manifest = dict(manifest["graphs"][graph_name])
        graph_manifest["path"] = str(output_path)
        graph_manifest["initializer_count"] = len(graph_entries)
        shared_graphs[graph_name] = graph_manifest
        initializers.extend(graph_entries)
        print(f"Created {output_path} with {len(graph_entries)} safetensor-backed initializers.")

    if all(graph_name in shared_graphs for graph_name in S2S_REQUIRED_GRAPH_FILES):
        merged_graph = create_resident_merged_s2s_model(output_dir, shared_graphs)
        shared_graphs[S2S_MERGED_GRAPH_NAME] = merged_graph
        alias_lookup = {
            (alias["graph"], alias["onnx_initializer"]): alias["merged_initializer"]
            for alias in merged_graph.get("initializer_aliases", [])
        }
        seen_merged_initializers: set[str] = set()
        for entry in list(initializers):
            if entry["graph"] not in S2S_REQUIRED_GRAPH_FILES:
                continue
            merged_name = alias_lookup.get((entry["graph"], entry["onnx_initializer"]), entry["onnx_initializer"])
            if merged_name in seen_merged_initializers:
                continue
            seen_merged_initializers.add(merged_name)
            merged_entry = dict(entry)
            merged_entry["graph"] = S2S_MERGED_GRAPH_NAME
            merged_entry["onnx_initializer"] = merged_name
            initializers.append(merged_entry)
        print(f"Created {merged_graph['path']} resident merged S2S graph.")
        capabilities = dict(capabilities or {})
        capabilities["resident_merged_graph"] = S2S_MERGED_GRAPH_NAME

    shared_manifest = {
        "bundle": bundle_name,
        "model_dir": str(model_dir),
        "opset": manifest["opset"],
        "dtype": manifest["dtype"],
        "hidden_size": manifest["hidden_size"],
        "audio_num_codebooks": manifest["audio_num_codebooks"],
        "vocab_size": manifest["vocab_size"],
        "initializer_binding": "safetensors_original_external_data",
        "graphs": shared_graphs,
        "initializers": initializers,
        "unique_source_tensors": sorted({entry["source_tensor"] for entry in initializers}),
    }
    if capabilities is not None:
        shared_manifest["capabilities"] = capabilities
    (output_dir / "shared_weights_manifest.json").write_text(json.dumps(shared_manifest, indent=2) + "\n", encoding="utf-8")

    if single_model_only:
        keep = {
            Path(shared_graphs[S2S_MERGED_GRAPH_NAME]["path"]).resolve(),
        }
    else:
        keep = {Path(graph["path"]).resolve() for graph in shared_graphs.values()}
    keep.add((output_dir / "shared_weights_manifest.json").resolve())
    for path in output_dir.iterdir():
        if path.is_file() and path.resolve() not in keep:
            path.unlink()
    return shared_manifest


def validate_shared_onnxruntime(shared_manifest: dict, model_dir: Path, validation_inputs: dict[str, dict[str, object]]) -> None:
    import onnxruntime as ort

    for graph_name, graph in shared_manifest["graphs"].items():
        path = Path(graph["path"])
        if not path.exists():
            print(f"Skipping ONNX Runtime validation for {graph_name}: {path} was intentionally omitted.")
            continue
        if graph_name not in validation_inputs:
            print(f"Skipping ONNX Runtime validation for {graph_name}: no validation inputs were exported.")
            continue
        graph_entries = [entry for entry in shared_manifest["initializers"] if entry["graph"] == graph_name]
        dtypes = sorted({str(entry["dtype"]) for entry in graph_entries})
        print(f"Python ORT shared validation for {graph_name} uses safetensors external-data spans ({dtypes}).")

        created_links: list[Path] = []
        try:
            for source_shard in sorted({str(entry["source_shard"]) for entry in graph_entries}):
                target_path = path.parent / source_shard
                if not target_path.exists():
                    ensure_hardlinked_safetensor(model_dir / source_shard, path.parent)
                    created_links.append(target_path)

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession(str(path), sess_options=session_options, providers=["CPUExecutionProvider"])
            outputs = session.run(None, validation_inputs[graph_name])
            print(f"ONNX Runtime shared {graph_name} outputs: {[str(output.shape) for output in outputs]}")
        finally:
            for target_path in created_links:
                target_path.unlink(missing_ok=True)
