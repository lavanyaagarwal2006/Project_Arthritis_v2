# Data

## RAM-H1200 (development)

- **Source:** https://huggingface.co/datasets/TokyoTechMagicYang/RAM-H1200-v1 (DOI 10.57967/hf/8548)
- **Paper:** Yang et al., "RAM-H1200: A Unified Evaluation and Dataset on Hand Radiographs for
  Rheumatoid Arthritis", NeurIPS 2026 Datasets & Benchmarks. arXiv:2605.05616
- **Contents:** 1,200 PA hand radiographs from 291 patients (241 RA, 50 non-RA) at 6 centers
  in Sapporo, Japan. Joint-level SvdH BE (16 joints) and JSN (15 joints) scores, BE masks,
  and bone segmentation. `Metadata.xlsx` has `isRA`, `Normalized PatientID` and `Center`.
- **License:** CC BY-NC-SA 4.0. Non-commercial use only, share-alike, attribution required.
  Commercial deployment needs separate permission or our own data.
- **Access:** gated. Accept the terms on the Hugging Face page, then run `huggingface-cli login`.

## RAM-W600 (not used for the primary analysis)

Wrist-only, with partially overlapping centers. Many of its non-RA images come from unrelated
public datasets (a pediatric hand atlas, fracture and tumor sets), so RA-vs-non-RA labels are
confounded with image source. We may use it later only for pretraining or wrist-joint
augmentation, and only after deduplicating it against RAM-H1200 by perceptual hash.

## External validation candidates

| Source | Status | Notes |
|---|---|---|
| Taipei Veterans General Hospital: "X-Ray Images of the Hands and Feet for RA" (Taiwan Medical AI and Data Portal, dataset D07) | to investigate | DICOM files plus scoring spreadsheets; access terms and patient counts unknown |
| ISIC New Delhi (Dr. Malviya) | needs ethics approval, a DUA and de-identification | the target deployment population |

## Never

- Never commit images, DICOMs or patient identifiers.
- Never mix RAM-W600 and RAM-H1200 without patient- or hash-level deduplication.
- Never touch the sealed test split before the analysis plan and the model are locked.
