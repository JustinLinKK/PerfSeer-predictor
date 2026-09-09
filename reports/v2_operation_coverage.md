# PerfSeer v2/v3 Operation Coverage

- Cases: 18
- Strict export success: 100.0%
- Validated non-strict success: 0.0%
- Complete graph success: 100.0%
- Legacy FX success: 22.2%
- Tensor nodes retained: 638/638
- Unique raw ATen/custom operations: 132
- Exact/family occurrence coverage: 99.2%

## Raw operations

| Operation | Occurrences | Output bytes |
| --- | ---: | ---: |
| `aten::__and__.Tensor` | 2 | 528 |
| `aten::_native_batch_norm_legit_no_training` | 56 | 1275392 |
| `aten::_to_copy` | 1 | 32 |
| `aten::abs` | 3 | 204 |
| `aten::adaptive_avg_pool1d` | 1 | 120 |
| `aten::adaptive_avg_pool2d` | 12 | 13824 |
| `aten::adaptive_avg_pool3d` | 1 | 576 |
| `aten::add.Tensor` | 29 | 290316 |
| `aten::amax` | 1 | 24 |
| `aten::amin` | 1 | 24 |
| `aten::arange` | 5 | 304 |
| `aten::arange.start` | 2 | 64 |
| `aten::argmax` | 1 | 48 |
| `aten::argmin` | 1 | 48 |
| `aten::as_strided` | 13 | 728 |
| `aten::avg_pool1d` | 1 | 144 |
| `aten::avg_pool2d` | 1 | 480 |
| `aten::avg_pool3d` | 1 | 1440 |
| `aten::bernoulli` | 1 | 96 |
| `aten::bilinear` | 1 | 16 |
| `aten::binary_cross_entropy_with_logits` | 1 | 4 |
| `aten::bmm` | 3 | 1536 |
| `aten::broadcast_tensors` | 1 | 224 |
| `aten::cat` | 3 | 384 |
| `aten::clamp` | 2 | 192 |
| `aten::clamp_min` | 1 | 96 |
| `aten::clone` | 9 | 23424 |
| `aten::conv1d` | 1 | 432 |
| `aten::conv2d` | 74 | 1287584 |
| `aten::conv3d` | 1 | 13440 |
| `aten::conv_transpose1d` | 1 | 288 |
| `aten::conv_transpose2d.input` | 1 | 1920 |
| `aten::conv_transpose3d.input` | 1 | 6720 |
| `aten::cos` | 1 | 96 |
| `aten::cross_entropy_loss` | 1 | 4 |
| `aten::div.Tensor` | 3 | 300 |
| `aten::einsum` | 1 | 96 |
| `aten::elu` | 1 | 96 |
| `aten::embedding` | 5 | 11200 |
| `aten::embedding_bag.padding_idx` | 1 | 160 |
| `aten::eq.Scalar` | 2 | 8 |
| `aten::erf` | 1 | 96 |
| `aten::exp` | 1 | 96 |
| `aten::expand` | 4 | 0 |
| `aten::gather` | 3 | 560 |
| `aten::ge.Scalar` | 1 | 16 |
| `aten::gelu` | 3 | 16480 |
| `aten::grid_sampler` | 1 | 720 |
| `aten::group_norm` | 1 | 1920 |
| `aten::gru.input` | 1 | 288 |
| `aten::hardsigmoid` | 9 | 9184 |
| `aten::hardswish` | 19 | 264704 |
| `aten::index.Tensor` | 8 | 72 |
| `aten::index_put` | 1 | 240 |
| `aten::index_select` | 3 | 80 |
| `aten::instance_norm` | 1 | 1920 |
| `aten::kl_div` | 1 | 4 |
| `aten::layer_norm` | 7 | 21440 |
| `aten::leaky_relu` | 1 | 96 |
| `aten::linalg_vector_norm` | 1 | 24 |
| `aten::linear` | 23 | 74360 |
| `aten::log` | 1 | 96 |
| `aten::log_softmax.int` | 1 | 112 |
| `aten::logsumexp` | 2 | 72 |
| `aten::lstm.input` | 1 | 336 |
| `aten::masked_fill.Scalar` | 3 | 272 |
| `aten::masked_select` | 1 | 0 |
| `aten::matmul` | 3 | 936 |
| `aten::max_pool1d` | 1 | 144 |
| `aten::max_pool2d` | 2 | 66016 |
| `aten::max_pool3d` | 1 | 1440 |
| `aten::maximum` | 1 | 96 |
| `aten::mean.dim` | 2 | 224 |
| `aten::minimum` | 1 | 96 |
| `aten::mish` | 1 | 96 |
| `aten::mm` | 1 | 48 |
| `aten::mse_loss` | 1 | 4 |
| `aten::mul.Tensor` | 17 | 93920 |
| `aten::mv` | 1 | 16 |
| `aten::native_dropout` | 1 | 120 |
| `aten::ne.Tensor` | 2 | 12 |
| `aten::new_ones` | 1 | 1 |
| `aten::new_zeros` | 4 | 352 |
| `aten::nll_loss_nd` | 1 | 4 |
| `aten::ones` | 2 | 0 |
| `aten::ones_like` | 1 | 240 |
| `aten::pad` | 1 | 3432 |
| `aten::permute` | 17 | 39808 |
| `aten::pow.Tensor_Scalar` | 2 | 32 |
| `aten::pow.Tensor_Tensor` | 1 | 96 |
| `aten::prod.dim_int` | 1 | 24 |
| `aten::rand_like` | 1 | 96 |
| `aten::relu` | 33 | 909984 |
| `aten::remainder.Tensor` | 1 | 96 |
| `aten::repeat` | 3 | 512 |
| `aten::rms_norm` | 1 | 320 |
| `aten::rnn_tanh.input` | 1 | 288 |
| `aten::rsqrt` | 1 | 96 |
| `aten::scaled_dot_product_attention` | 3 | 8832 |
| `aten::scatter.src` | 1 | 240 |
| `aten::scatter_add` | 4 | 352 |
| `aten::scatter_reduce.two` | 1 | 240 |
| `aten::select.int` | 16 | 2152 |
| `aten::selu` | 1 | 96 |
| `aten::sigmoid` | 2 | 192 |
| `aten::silu` | 2 | 1248 |
| `aten::sin` | 1 | 96 |
| `aten::softmax.int` | 3 | 1112 |
| `aten::softplus` | 1 | 96 |
| `aten::sort` | 1 | 720 |
| `aten::split_with_sizes` | 2 | 384 |
| `aten::sqrt` | 1 | 96 |
| `aten::squeeze.dims` | 5 | 2304 |
| `aten::stack` | 1 | 384 |
| `aten::std.dim` | 1 | 24 |
| `aten::sub.Tensor` | 1 | 96 |
| `aten::sum.dim_IntList` | 1 | 8 |
| `aten::tanh` | 1 | 96 |
| `aten::tile` | 1 | 384 |
| `aten::topk` | 2 | 720 |
| `aten::upsample_bilinear2d.vec` | 1 | 7680 |
| `aten::var.dim` | 1 | 24 |
| `aten::view` | 2 | 4352 |
| `aten::where.self` | 1 | 96 |
| `aten::zeros` | 4 | 192 |
| `aten::zeros_like` | 2 | 480 |
| `prim::getitem` | 78 | 1278320 |
| `prims::broadcast_in_dim` | 15 | 7984 |
| `prims::collapse_view` | 9 | 11776 |
| `prims::split_dim` | 13 | 30496 |
| `prims::transpose` | 3 | 576 |
| `prims::view_of` | 3 | 192 |

Report SHA-256: `bcc207f6c9a7cb4ac4c88631e914f59ae02245b57b31caf5e33b1b09d17b49e9`
