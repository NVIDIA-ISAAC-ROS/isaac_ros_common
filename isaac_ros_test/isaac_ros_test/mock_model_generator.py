# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate a mock model with empty weights for testing DNN pipeline structure."""

import argparse
import functools
import operator
from pathlib import Path
from typing import List, NamedTuple, Tuple

import torch


class MockModelGenerator:

    class Binding(NamedTuple):
        name: str
        shape: Tuple[int, ...]
        datatype: torch.dtype

    class MockModel(torch.nn.Module):
        def __init__(
            self,
            input_bindings: List['MockModelGenerator.Binding'],
            output_bindings: List['MockModelGenerator.Binding'],
            intermediate_size: int = 4,
        ):
            super().__init__()
            self.input_bindings = input_bindings
            self.output_bindings = output_bindings

            # Precompute the output sizes for reshaping merged layer output
            # Handle dynamic batch sizes (dim = -1) by taking the absolute value of the product
            self.output_sizes = [
                abs(functools.reduce(operator.mul, b.shape)) for b in self.output_bindings
            ]

            total_input_elements = sum(
                abs(functools.reduce(operator.mul, b.shape)) for b in self.input_bindings)
            total_output_elements = sum(self.output_sizes)

            self.layers = torch.nn.Sequential(
                torch.nn.Linear(total_input_elements, intermediate_size),
                torch.nn.Linear(intermediate_size, total_output_elements)
            )

        def forward(self, *inputs):
            # Cast, flatten, and concatenate the input tensors to a single tensor
            merged_input = torch.cat(
                [t.float().flatten() for t in inputs],
            )

            # Run the forward pass through the merged layer
            merged_output = self.layers(merged_input)

            output_tensors = []
            start_idx = 0
            for binding, size in zip(self.output_bindings, self.output_sizes):
                end_idx = start_idx + size

                # Slice the merged output to get the current output binding
                output_tensor = merged_output[start_idx:end_idx]\
                    .reshape(binding.shape)\
                    .to(binding.datatype)

                output_tensors.append(output_tensor)

                # Update the start index for the next output binding
                start_idx = end_idx

            return output_tensors

    @classmethod
    def generate(
        cls,
        input_bindings: List[Binding],
        output_bindings: List[Binding],
        output_onnx_path: Path
    ):
        model = cls.MockModel(input_bindings, output_bindings)

        # Generate dummy input tensors
        dummy_input = [
            # Use 1 as the default dimension for dynamic axes
            torch.ones([d if d != -1 else 1 for d in b.shape]).to(b.datatype)
            for b in input_bindings
        ]

        # Identify dynamic axes across bindings
        dynamic_axes = {}
        for binding in (*input_bindings, *output_bindings):
            dynamic_axes[binding.name] = {
                i: f'dynamic_{i}' for i, size in enumerate(binding.shape)
                if size == -1
            }

        torch.onnx.export(
            model,
            tuple(dummy_input),
            output_onnx_path,
            input_names=[binding.name for binding in input_bindings],
            output_names=[binding.name for binding in output_bindings],
            dynamic_axes=dynamic_axes
        )


def parse_bindings(bindings_str):
    bindings = []
    for binding_str in bindings_str.split(','):
        name, shape_str, datatype_str = binding_str.split(':')
        shape = tuple(map(int, shape_str.split('x')))
        datatype = getattr(torch, datatype_str)
        bindings.append(MockModelGenerator.Binding(name, shape, datatype))
    return bindings


def main(input_bindings_str, output_bindings_str, output_onnx_path):
    input_bindings = parse_bindings(input_bindings_str)
    output_bindings = parse_bindings(output_bindings_str)

    MockModelGenerator.generate(input_bindings, output_bindings, output_onnx_path)


# Example usage:
# images:-1x3x640x640:float32,orig_target_sizes:-1x2:int64
# labels:-1x300:float32,boxes:-1x300x4:float32,scores:-1x300:float32

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Generate a mock model with empty weights for testing DNN pipeline structure.')
    ap.add_argument(
        '--input-bindings', type=str, required=True,
        help='Input bindings in the format name:shape:datatype,name:shape:datatype,... '
             'Use -1 for dynamic axes.'
    )
    ap.add_argument(
        '--output-bindings', type=str, required=True,
        help='Output bindings in the format name:shape:datatype,name:shape:datatype,... '
             'Use -1 for dynamic axes.'
    )
    ap.add_argument(
        'output_onnx_path', type=Path,
        help='Path to save the generated ONNX model.'
    )

    args = ap.parse_args()
    main(args.input_bindings, args.output_bindings, args.output_onnx_path)
