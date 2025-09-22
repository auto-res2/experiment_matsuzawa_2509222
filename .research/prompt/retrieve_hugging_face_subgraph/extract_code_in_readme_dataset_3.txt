
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{}
---

# Dataset Card for CIFAR-100-C

<!-- Provide a quick summary of the dataset. -->

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->
The CIFAR-100-C dataset is an extension of CIFAR-100 designed to evaluate model robustness to common corruptions. It consists of 950,000 images derived from the original CIFAR-100 test set (10,000 images) by applying 19 different corruption types at 5 severity levels. The corruptions include noise, blur, weather effects, and digital distortions. This dataset is widely used for benchmarking robustness in image classification tasks.

### Dataset Sources

<!-- Provide the basic links for the dataset. -->

- **Homepage:** https://github.com/hendrycks/robustness
- **Paper:** Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261.

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

Each sample in the dataset contains:

- **image**: A 32×32 RGB image in PNG format

- **label**: An integer between 0 and 99, representing the class

- **corruption_name**: The name of the applied corruption

- **corruption_level**: An integer between 1 and 5 indicating severity


Total images: 950,000

Classes: 100 (e.g., apple, aquarium fish, baby, bear, beaver, bed, bee, beetle, bicycle, bottle, etc.)

Corruptions: 19 types (e.g., Gaussian noise, motion blur, contrast, fog, frost, elastic transform, pixelate, JPEG compression, etc.)

Severity Levels: 5 (ranging from least to most severe)

Splits:

- **Test:** 950,000 images

Image specs: PNG format, 32×32 pixels, RGB

## Example Usage
Below is a quick example of how to load this dataset via the Hugging Face Datasets library.
```
from datasets import load_dataset  

# Load the dataset  
dataset = load_dataset("randall-lab/cifar100-c", split="test", trust_remote_code=True)  

# Access a sample from the dataset  
example = dataset[0]  
image = example["image"]  
label = example["label"]  

image.show()  # Display the image  
print(f"Label: {label}")
```

## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

@article{hendrycks2019benchmarking,
  title={Benchmarking neural network robustness to common corruptions and perturbations},
  author={Hendrycks, Dan and Dietterich, Thomas},
  journal={arXiv preprint arXiv:1903.12261},
  year={2019}
}

Output:
{
    "extracted_code": "from datasets import load_dataset\n\ndataset = load_dataset(\"randall-lab/cifar100-c\", split=\"test\", trust_remote_code=True)\n\nexample = dataset[0]\nimage = example[\"image\"]\nlabel = example[\"label\"]\n\nimage.show()\nprint(f\"Label: {label}\")"
}
