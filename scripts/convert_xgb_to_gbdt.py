#!/usr/bin/env python3
"""
Convert XGBoost model.json to fraudctl GBDT binary format v2.

Usage:
    python3 convert_xgb_to_gbdt.py <model.json> <output.bin>

Reads an XGBoost model trained with reg:squarederror (regression, no sigmoid)
and writes a compact binary file readable by internal/gbdt.Load().

Binary format v2 (little-endian):
    magic       [4]byte  "GBDT"
    version     uint32   = 2
    numTrees    uint32
    numFeatures uint32
    initPred    float32
    sigmoid     uint8    0=no, 1=yes
    pad         [3]byte
    per-tree:
      numNodes  uint32
      per-node (12 bytes):
        feat    uint8    feature index (0..13)
        pad     uint8
        left    uint16   left child index (0 for leaf)
        right   uint16   right child index (0 for leaf)
        leaf    uint8    1 if leaf node
        pad     uint8
        value   float32  threshold for internal nodes, leaf value for leaves
"""

import json
import struct
import sys


def convert(json_path: str, out_path: str) -> None:
    with open(json_path) as f:
        m = json.load(f)

    learner = m["learner"]
    obj = learner["objective"]["name"]
    base_score = float(learner["learner_model_param"]["base_score"])
    trees = learner["gradient_booster"]["model"]["trees"]

    sigmoid = 1 if ("logistic" in obj or "binary" in obj) else 0
    num_trees = len(trees)
    num_features = 14

    print(
        f"objective={obj}, sigmoid={bool(sigmoid)}, "
        f"base_score={base_score:.6f}, trees={num_trees}"
    )

    buf = bytearray()

    # Header: magic(4) version(4) numTrees(4) numFeatures(4) initPred(4) sigmoid(1) pad(3)
    buf += b"GBDT"
    buf += struct.pack("<I", 2)            # version
    buf += struct.pack("<I", num_trees)
    buf += struct.pack("<I", num_features)
    buf += struct.pack("<f", base_score)
    buf += struct.pack("<B", sigmoid)
    buf += b"\x00\x00\x00"                # pad

    for t in trees:
        n = int(t["tree_param"]["num_nodes"])
        left = t["left_children"]
        right = t["right_children"]
        split_idx = t["split_indices"]
        split_val = t["split_conditions"]

        buf += struct.pack("<I", n)

        for i in range(n):
            is_leaf = left[i] == -1
            feat = split_idx[i] if not is_leaf else 0
            lc = left[i] if not is_leaf else 0
            rc = right[i] if not is_leaf else 0

            # thresh_or_value: split threshold for internal nodes,
            #                  leaf value for leaf nodes.
            val = float(split_val[i])

            # 12 bytes per node:
            # feat(1) pad(1) left(2) right(2) leaf(1) pad(1) value(4)
            buf += struct.pack("<B", feat & 0xFF)
            buf += b"\x00"
            buf += struct.pack("<H", lc & 0xFFFF)
            buf += struct.pack("<H", rc & 0xFFFF)
            buf += struct.pack("<B", 1 if is_leaf else 0)
            buf += b"\x00"
            buf += struct.pack("<f", val)

    with open(out_path, "wb") as f:
        f.write(buf)

    print(f"Written {len(buf):,} bytes → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <model.json> <output.bin>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
