# Computer-Vision Model Architecture Families in PerfSeer-predictor

- `fields` are the architecture knobs each family exposes to the generator; `assertions` are the operations every generated instance must contain; `adapter_ids` are the MLE-bench Lite datasets the family is bound to.

| family_id | architecture fields | required ops | bound datasets |
|---|---|---|---|
| `resnet50` | `width_multiplier`, `stage_depths`, `input_resolution`, `act`| `conv.2d`, `batch_norm`, `add.Tensor` | histopathologic-cancer, dogs-vs-cats |
| `efficientnet_b0_b4` | `width_multiplier`, `depth_multiplier`, `input_resolution` | `conv.2d`, `silu` | dog-breed, plant-pathology |
| `mobilenet_v3_large` | `width_multiplier`, `input_resolution` | `conv.2d`, `hardswish` | aerial-cactus, dog-breed |
| `inception_v3` | `aux_logits`, `input_resolution` | `conv.2d`, `cat` | dogs-vs-cats, aptos2019 |
| `vit_s16` | `depth`, `heads`, `patch_size`, `input_resolution` | `scaled_dot_product_attention`, `layer_norm` | siim-isic-melanoma, ranzcr-clip |
| `swin_t` | `window_size`, `depths`, `heads`, `input_resolution` | `scaled_dot_product_attention`, `structural:roll`, `layer_norm` | siim-isic-melanoma, ranzcr-clip |
| `mlp_mixer_s` | `depth`, `channel_dim`, `token_dim`, `patch_size` | `linear`, `gelu` | leaf-classification, dog-breed |
| `stn_cnn` | `localization_width`, `input_resolution` | `grid_sampler`, `conv.2d` | aerial-cactus, dogs-vs-cats |
| `unet` | `depth`, `base_channels`, `input_resolution`, `norm` | `conv.2d`, `upsample.bilinear2d` | denoising-dirty-documents |
| `pix2pix` | `generator_width`, `discriminator_width`, `input_resolution` | `conv.2d`, `conv_transpose.2d`, `binary_cross_entropy_with_logits` | denoising-dirty-documents |
|`restormer` | `depths`, `heads`, `width`, `input_resolution` | `conv.2d`, `scaled_dot_product_attention`, `layer_norm` | denoising-dirty-documents |

## Operator-substitution variant axes

- Variants are no longer only depth/width rescalings. Each family is also expanded by substituting operator blocks, so the corpus covers modern block designs at fixed macro-architecture.

| Substitution | Original block | Replacement | Applies to |
|---|---|---|---|
| Gated feed-forward | MLP or pointwise `conv1x1` expansion | SwiGLU | every family with an MLP or inverted-bottleneck expansion: `vit_s16`, `swin_t`, `mlp_mixer_s`, `restormer`, `efficientnet_b0_b4`, `mobilenet_v3_large` |
| Modernized convolution | channel-constant `Conv2d` stack | ConvNeXt block, V1 for small models and V2 with global response normalization for large models | `resnet50`, `mish_resnet`, `prelu_elu_cnn`, `vgg_cnn`, `unet_groupnorm` encoder |
| Upsampling path | `ConvTranspose2d` | `Conv2d` followed by `Upsample`, or `Conv2d` followed by `PixelShuffle` | `pix2pix`, `unet_groupnorm`, `restormer` decoder, any segmentation or image-to-image decoder |
| Attention layout | multi-head attention | multi-head latent attention (MLA) or grouped-query attention (GQA) | `vit_s16`, `swin_t`, `restormer` |

- These substitutions are exactly the cases where analytic cost models drift from measurement, which is why they belong in the training corpus.
