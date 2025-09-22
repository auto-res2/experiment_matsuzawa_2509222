
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype: int64
  - name: classname
    dtype: string
  - name: relpath
    dtype: string
  splits:
  - name: test
    num_bytes: 2214054103.5
    num_examples: 7500
  download_size: 2213438358
  dataset_size: 2214054103.5
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---

Output:
{
    "extracted_code": ""
}
