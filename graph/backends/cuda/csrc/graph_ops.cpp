// Copyright (C) 2022 COAI @ Tsinghua University
// SPDX-License-Identifier: Apache-2.0
// Python bindings for the optional DA-DLM CUDA graph backend.

#include <torch/extension.h>

#include <tuple>


std::tuple<torch::Tensor, torch::Tensor> dag_loss(
    const torch::Tensor& emission_scores,
    const torch::Tensor& compact_transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths,
    bool require_gradient,
    int config);

std::tuple<torch::Tensor, torch::Tensor> dag_loss_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& alpha,
    const torch::Tensor& beta,
    const torch::Tensor& emission_scores,
    const torch::Tensor& compact_transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths,
    int emission_config,
    int transition_config);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
alignment_posterior_forward(
    const torch::Tensor& emission_scores,
    const torch::Tensor& compact_transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths,
    int forward_config);

std::tuple<torch::Tensor, torch::Tensor> alignment_posterior_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& posterior,
    const torch::Tensor& alpha,
    const torch::Tensor& beta,
    const torch::Tensor& emission_scores,
    const torch::Tensor& compact_transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths,
    int backward_config);

std::tuple<torch::Tensor, torch::Tensor> viterbi_alignment(
    const torch::Tensor& emission_scores,
    const torch::Tensor& compact_transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths,
    int config);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> viterbi_path(
    const torch::Tensor& node_scores,
    const torch::Tensor& transition_scores,
    const torch::Tensor& node_lengths,
    int config);

std::tuple<torch::Tensor, torch::Tensor> sample_path(
    const torch::Tensor& transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& uniforms,
    int forward_config,
    int sampling_config);

torch::Tensor sample_alignment(
    const torch::Tensor& emission_scores,
    const torch::Tensor& compact_transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths,
    const torch::Tensor& uniforms,
    int forward_config,
    int sampling_config);

std::tuple<torch::Tensor, torch::Tensor> path_partition_forward(
    const torch::Tensor& transition_scores,
    const torch::Tensor& node_lengths,
    int config);

torch::Tensor path_partition_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& alpha,
    const torch::Tensor& transition_scores,
    const torch::Tensor& node_lengths,
    int config);

torch::Tensor path_suffix_partition(
    const torch::Tensor& transition_scores,
    const torch::Tensor& node_lengths,
    int config);

torch::Tensor logsoftmax_gather(
    torch::Tensor workspace,
    const torch::Tensor& select_idx,
    bool require_gradient,
    int threads);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "alignment_partition_forward",
      &dag_loss,
      "Fixed-target alignment partition forward");
  module.def(
      "alignment_partition_backward",
      &dag_loss_backward,
      "Fixed-target alignment partition backward");
  module.def(
      "alignment_posterior_forward",
      &alignment_posterior_forward,
      "Fixed-target alignment log-posterior forward");
  module.def(
      "alignment_posterior_backward",
      &alignment_posterior_backward,
      "Fixed-target alignment log-posterior arbitrary VJP");
  module.def(
      "path_partition_forward",
      &path_partition_forward,
      "All-path transition partition forward");
  module.def(
      "path_partition_backward",
      &path_partition_backward,
      "All-path transition partition backward");
  module.def(
      "path_suffix_partition",
      &path_suffix_partition,
      "All-path suffix log weights");
  module.def(
      "viterbi_alignment",
      &viterbi_alignment,
      "Fixed-target Viterbi alignment");
  module.def(
      "viterbi_path",
      &viterbi_path,
      "Maximum-score complete DAG path");
  module.def(
      "sample_path",
      &sample_path,
      "All-path FFBS sampling with caller-provided uniforms");
  module.def(
      "sample_alignment",
      &sample_alignment,
      "Fixed-target FFBS sampling with caller-provided uniforms");
  module.def(
      "logsoftmax_gather",
      &logsoftmax_gather,
      "Rowwise log-softmax gather with optional in-place softmax workspace");
}
