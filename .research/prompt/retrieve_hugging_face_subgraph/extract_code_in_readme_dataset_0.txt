
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: labels
    dtype: int64
  - name: class_ids
    dtype: string
  - name: class_names
    dtype: string
  splits:
  - name: train
    num_bytes: 154860034195.012
    num_examples: 1281167
  download_size: 146539410924
  dataset_size: 154860034195.012
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

Output:
{
    "extracted_code": ""
}
