#!/usr/bin/env python3
"""
Merge feature_runner.onnx and post_runner.onnx into a single
foundation_stereo.onnx by inserting a BuildGwcVolume custom-op node
between the two sub-graphs.

Works at the raw ONNX protobuf level to avoid graphsurgeon tensor-name
conflicts between the two graphs.

Usage:
    python merge_onnx.py \
        --feature_onnx feature_runner.onnx \
        --post_onnx     post_runner.onnx \
        --output_onnx   foundation_stereo.onnx \
        --maxdisp 48 --num_groups 8
"""
import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def _prefix_graph(graph: onnx.GraphProto, prefix: str,
                  keep_names: set) -> onnx.GraphProto:
    """
    Return a copy of *graph* where every internal tensor name is prefixed.
    Tensor names in *keep_names* (graph I/O) are left unchanged.
    Node names are also prefixed to avoid duplicates.
    """
    # Build rename map for all tensors that appear in the graph
    all_names: set[str] = set()
    for n in graph.node:
        all_names.update(n.input)
        all_names.update(n.output)
    for init in graph.initializer:
        all_names.add(init.name)

    rename = {
        name: name if (name in keep_names or name == "") else (prefix + name)
        for name in all_names
    }

    new_graph = onnx.GraphProto()

    # Nodes
    for node in graph.node:
        new_node = onnx.NodeProto()
        new_node.CopyFrom(node)
        new_node.name = prefix + node.name if node.name else ""
        del new_node.input[:]
        del new_node.output[:]
        new_node.input.extend(rename[i] for i in node.input)
        new_node.output.extend(rename[o] for o in node.output)
        new_graph.node.append(new_node)

    # Initializers (weights)
    for init in graph.initializer:
        new_init = onnx.TensorProto()
        new_init.CopyFrom(init)
        new_init.name = rename[init.name]
        new_graph.initializer.append(new_init)

    # value_info (intermediate shape annotations)
    for vi in graph.value_info:
        new_vi = onnx.ValueInfoProto()
        new_vi.CopyFrom(vi)
        new_vi.name = rename.get(vi.name, vi.name)
        new_graph.value_info.append(new_vi)

    # Inputs / outputs keep their original names (they are in keep_names)
    new_graph.input.extend(graph.input)
    new_graph.output.extend(graph.output)

    return new_graph


def merge_ffs_onnx(
    feature_onnx: str,
    post_onnx: str,
    output_onnx: str,
    maxdisp: int,
    num_groups: int,
) -> None:
    feat_m = onnx.load(feature_onnx)
    post_m = onnx.load(post_onnx)

    feat_g = feat_m.graph
    post_g = post_m.graph

    # Names that must NOT be renamed in the feature graph:
    #  - its own inputs (left, right) — become the merged graph's inputs
    #  - its own outputs (features_*) — become connections to post graph & GWC
    feat_input_names  = {t.name for t in feat_g.input}
    feat_output_names = {t.name for t in feat_g.output}
    post_input_names  = {t.name for t in post_g.input}
    keep_names = feat_input_names | feat_output_names | post_input_names

    # Rename feature-graph internals to avoid name conflicts with post-graph
    feat_g_r = _prefix_graph(feat_g, "feat/", keep_names=keep_names)

    # Build the GWC plugin node
    # Inputs:  features_left_04, features_right_04  (fp32 from feature runner)
    # Output:  gwc_volume  (fp16, fed to post runner)
    # shape of gwc_volume: (1, num_groups, maxdisp, H/4, W/4)
    # -- find H/4, W/4 from feat output shape annotation
    feat04_shape = None
    for vi in feat_g.output:
        if vi.name == "features_left_04":
            d = vi.type.tensor_type.shape.dim
            feat04_shape = [x.dim_value for x in d]   # (1, C, H4, W4)
            break

    gwc_shape = [1, num_groups, maxdisp,
                 feat04_shape[2] if feat04_shape else 0,
                 feat04_shape[3] if feat04_shape else 0]

    gwc_vi = helper.make_tensor_value_info(
        "gwc_volume", TensorProto.FLOAT16, gwc_shape
    )

    gwc_node = helper.make_node(
        op_type="BuildGwcVolume",
        inputs=["features_left_04", "features_right_04"],
        outputs=["gwc_volume"],
        name="GwcVolumePlugin",
        maxdisp=maxdisp,
        num_groups=num_groups,
    )

    # ------------------------------------------------------------------ #
    # Build merged graph
    # ------------------------------------------------------------------ #
    # Graph inputs: left, right  (from feature_runner)
    # Graph outputs: disp        (from post_runner)
    merged_inputs  = list(feat_g_r.input)   # left, right
    merged_outputs = list(post_g.output)    # disp

    all_nodes = list(feat_g_r.node) + [gwc_node] + list(post_g.node)
    all_inits = list(feat_g_r.initializer) + list(post_g.initializer)

    # value_info: combine feat (renamed) + gwc_vi + post
    all_value_info = (
        list(feat_g_r.value_info)
        + list(feat_g_r.output)   # feat outputs become internal
        + [gwc_vi]
        + [vi for vi in post_g.input  if vi.name != "gwc_volume"]
        + list(post_g.value_info)
    )

    merged_graph = helper.make_graph(
        all_nodes,
        "foundation_stereo",
        merged_inputs,
        merged_outputs,
        initializer=all_inits,
    )
    merged_graph.value_info.extend(all_value_info)

    opset = onnx.OperatorSetIdProto()
    opset.version = 17
    merged_model = helper.make_model(merged_graph, opset_imports=[opset])
    merged_model.ir_version = feat_m.ir_version

    # Copy external initializer data (needed for large ONNX files)
    onnx.save(merged_model, output_onnx)
    print(f"Merged ONNX saved → {output_onnx}")
    print(f"  inputs:  {[t.name for t in merged_inputs]}")
    print(f"  outputs: {[t.name for t in merged_outputs]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_onnx", required=True)
    parser.add_argument("--post_onnx",    required=True)
    parser.add_argument("--output_onnx",  required=True)
    parser.add_argument("--maxdisp",    type=int, default=48)
    parser.add_argument("--num_groups", type=int, default=8)
    args = parser.parse_args()
    merge_ffs_onnx(
        args.feature_onnx, args.post_onnx, args.output_onnx,
        args.maxdisp, args.num_groups,
    )
