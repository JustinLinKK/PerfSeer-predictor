# PerfSeer v3 model-corpus compatibility matrix

This report intentionally separates structural encoding from exact registry identity and measured prediction accuracy.

## Summary

- Structurally encodable captured models: 18/18.
- Models whose every observed operation has an exact registry ID: 0/18.
- Accuracy-validated models: 0/18.
- Measured profiler-time unknown/custom fraction: 0.0000%.

Accuracy support remains unvalidated until grouped production scheduler labels and held-out predictions exist.

## Model corpus

| Case | Source family | Strict | Structural | Exact-only | Exact fraction | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| smoke_residual_conv | residual_cnn | yes | yes | no | 60.0% | no |
| smoke_sequence_attention | transformer_block | yes | yes | no | 90.9% | no |
| smoke_index_reduction | index_reduction | yes | yes | no | 80.0% | no |
| p0_dense_variants | dense_matrix | yes | yes | no | 28.6% | no |
| p0_convolution_variants | convolution | yes | yes | no | 16.7% | no |
| p0_attention_variants | attention | yes | yes | no | 8.6% | no |
| p0_normalization_variants | normalization | yes | yes | no | 16.7% | no |
| p0_elementwise_unary_variants | elementwise_unary | yes | yes | no | 15.2% | no |
| p0_reduction_loss_variants | reduction_loss | yes | yes | no | 13.6% | no |
| p0_pool_resample_variants | pool_resample | yes | yes | no | 0.0% | no |
| p0_layout_variants | layout_shape | yes | yes | no | 0.0% | no |
| p0_index_scatter_variants | index_scatter | yes | yes | no | 9.1% | no |
| p0_embedding_sequence_variants | embedding_sequence | yes | yes | no | 5.9% | no |
| p0_training_random_variants | random_regularization | yes | yes | no | 0.0% | no |
| source_torchvision_resnet18 | residual_cnn | yes | yes | no | 32.6% | no |
| source_torchvision_mobilenet_v3_small | mobile_cnn | yes | yes | no | 31.6% | no |
| source_transformers_small_bert | transformer_encoder | yes | yes | no | 38.2% | no |
| source_pyg_small_gcn | graph_convolution | yes | yes | no | 6.0% | no |

## Operation families

| Family | Structural path | Rules | Exact IDs | Observed nodes | Measured GPU ms | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| unknown_or_custom | generic_hash | 0 | 0 | 10 | 0.000 | no |
| dense_matrix | family_embedding | 9 | 2 | 33 | 328.522 | no |
| convolution | family_embedding | 7 | 1 | 79 | 544.978 | no |
| attention | family_embedding | 4 | 0 | 3 | 47.663 | no |
| normalization | family_embedding | 7 | 2 | 66 | 265.995 | no |
| elementwise | family_embedding | 15 | 2 | 64 | 279.399 | no |
| activation_unary | family_embedding | 22 | 2 | 87 | 313.084 | no |
| reduction | family_embedding | 12 | 1 | 13 | 248.303 | no |
| loss_probability | family_embedding | 8 | 1 | 9 | 49.387 | no |
| pool_resample | family_embedding | 15 | 0 | 24 | 7.176 | no |
| layout_shape | family_embedding | 35 | 1 | 194 | 578.805 | no |
| index_scatter | family_embedding | 14 | 2 | 44 | 357.486 | no |
| embedding_sequence | family_embedding | 6 | 1 | 9 | 168.309 | no |
| random_regularization | family_embedding | 4 | 0 | 3 | 3.699 | no |
| sparse_graph | family_embedding | 0 | 0 | 0 | 0.000 | no |
| quantized_low_precision | family_embedding | 0 | 0 | 0 | 0.000 | no |
| spectral_linalg | family_embedding | 0 | 0 | 0 | 0.000 | no |
| custom_fused | family_embedding | 0 | 0 | 0 | 0.000 | no |
| training | family_embedding | 3 | 0 | 0 | 8.856 | no |
| optimizer | family_embedding | 18 | 0 | 0 | 0.000 | no |

Report SHA-256: `720e7ec3e51efbcea730f3f34570397fe50dcc65775b3cff25cdb43d7cbd9d99`
