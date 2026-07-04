# Dataset Source Approval Registry

This directory tracks candidate real datasets for the scheduler-grade PerfSeer
workflow. Raw data, extracted archives, prepared tensors, and downloaded
competition files do not belong in git.

Candidates are split into two execution tiers:

- `local`: default RTX 5090 workstation flow. These candidates are selected to
  stay under roughly a 30 GB local working budget when possible.
- `nautilus`: PVC-backed Nautilus flow for very large or archival reference
  datasets. Local downloads are refused unless `--allow-nautilus-only` is
  passed explicitly.

Workflow:

1. Review one Markdown file under `candidates/`.
2. If the source is acceptable, mark that dataset `approved` in
   `registry.json` or run:

   ```bash
   python scripts/manage_dataset_sources.py approve <dataset_id>
   ```

3. List and download only approved local datasets:

   ```bash
   python scripts/manage_dataset_sources.py list --tier local
   python scripts/manage_dataset_sources.py download <dataset_id> --raw-root datasets/raw
   ```

4. Prepare deterministic subset masks and real-derived metadata:

   ```bash
   python scripts/manage_dataset_sources.py prepare <dataset_id> \
     --raw-root datasets/raw \
     --prepared-root datasets/prepared
   ```

The download command intentionally fails for `pending` datasets and for
`nautilus` tier datasets on the local path. Kaggle sources also require
accepting the competition rules on the Kaggle website and setting up Kaggle
credentials.

Local primary candidates:

| Task family | Local dataset id | Larger/reference tier |
| --- | --- | --- |
| image classification | `cassava_leaf_disease` | `imagenet_object_localization` |
| image segmentation | `pothole_image_segmentation` | `carvana_image_masking` |
| object detection | `taco_yolo_object_detection` | `global_wheat_detection` |
| text classification | `jigsaw_toxic_comment` | none |
| seq2seq text | `cnn_dailymail_summarization` | none |
| audio classification | `animal_audio_classification` | `birdclef_2024` |
| time series | `store_sales_time_series` | none |
| tabular | `credit_card_default` | none |
| graph | `ogbn_products` | none |

For Nautilus/PVC runs only:

```bash
python scripts/manage_dataset_sources.py list --tier nautilus
python scripts/manage_dataset_sources.py download <dataset_id> \
  --raw-root /pvc/datasets/raw \
  --allow-nautilus-only
```
