# EfficientQAT

[EfficientQAT](https://arxiv.org/abs/2407.11062) is a novel quantization technical, which pushes the limitation of uniform (INT) quantization in an efficient manner. Due to the leverage of standard INT quantization, the quantized model of EfficientQAT can also be transferred into other formats, such as GPTQ, BitBLAS, etc.  


In this repo, we provide three type checkpoints, one is EQAT, indicats the original checkpoints of EfficientQAT. The other two are GPTQ and BitBLAS respectively.


## Model Zoo

We provide a number of prequantized EfficientQAT models as follows: 

- WikiText2 PPL is measured in 2048 context length.
- Avg. Accuracy indicate the average accuracy in 5 zero-shot reasoning tasks (WinoGrande,PIQA,HellaSwag,Arc-Easy, Arc-Challenge) with [lm-eval v0.4.2](https://github.com/EleutherAI/lm-evaluation-harness).
- 1GB = $10^9$ Bit
- Hub Link: EQAT indicates the original checkpoints. We also transfer the checkpoints into GPTQ and BitBLAS formats, which can be loaded directly through [GPTQModel](https://github.com/ModelCloud/GPTQModel). (PS: [GPTQModel](https://github.com/ModelCloud/GPTQModel) is a official bug-fixed repo of AutoGPTQ, which would be merged into [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) in future.)

| Model | Quantization | WikiText2 PPL | Avg. Accuracy | Model Size (GB) | Hub link|
|-------|--------------|---------------|---------------|-----------------|----------|
Llama-2-7B|fp16|5.47|64.86|13.2|-|
Llama-2-7B|w4g128|5.53|64.27|3.7|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w4g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-2-7b-EfficientQAT-w4g128-BitBLAS)|
Llama-2-7B|w3g128|5.81|64.02|3.1|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w3g128)|
Llama-2-7B|w2g64|6.86|60.14|2.3|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w2g64)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w2g64-GPTQ)\|[BitBLAS](Llama-2-7b-EfficientQAT-w2g64-BitBLAS)|
Llama-2-7B|w2g128|7.17|59.50|2.2|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w2g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w2g128-GPTQ)\|[BitBLAS](Llama-2-7b-EfficientQAT-w2g128-BitBLAS)|
Llama-2-13B|fp16|4.88|67.81|25.4|-|
Llama-2-13B|w4g128|4.93|67.52|6.8|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w4g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-2-7b-EfficientQAT-w4g128-BitBLAS)|
Llama-2-13B|w3g128|5.12|67.28|5.6|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w3g128)|
Llama-2-13B|w2g64|5.96|64.88|4.0|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w2g64)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w2g64-GPTQ)\|[BitBLAS](Llama-2-13b-EfficientQAT-w2g64-BitBLAS)|
Llama-2-13B|w2g128|6.08|63.88|3.8|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w2g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w2g128-GPTQ)\|[BitBLAS](Llama-2-13b-EfficientQAT-w2g128-BitBLAS)|
Llama-2-70B|fp16|3.32|72.41|131.6|-|
Llama-2-70B|w4g128|3.39|72.62|35.8|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w4g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-2-70b-EfficientQAT-w4g128-BitBLAS)|
Llama-2-70B|w3g128|3.61|71.76|29.1|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w3g128)|
Llama-2-70B|w2g64|4.52|69.48|20.1|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w2g64)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w2g64-GPTQ)\|[BitBLAS](Llama-2-70b-EfficientQAT-w2g64-BitBLAS)|
Llama-2-70B|w2g128|4.61|68.93|18.9|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w2g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w2g128-GPTQ)\|[BitBLAS](Llama-2-70b-EfficientQAT-w2g128-BitBLAS)|
Llama-3-8B|fp16|6.14|68.58|13.0|-|
Llama-3-8B|w4g128|6.47|68.43|5.4|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w4g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-3-8b-EfficientQAT-w4g128-BitBLAS)|
Llama-3-8B|w3g128|7.09|67.35|4.7|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w3g128)|
Llama-3-8B|w2g64|9.41|60.76|3.9|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w2g64)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-3-8b-EfficientQAT-w2g64-BitBLAS)|
Llama-3-8B|w2g128|9.80|59.36|3.8|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w2g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w2g128-GPTQ)\|[BitBLAS](Llama-3-8b-EfficientQAT-w2g128-BitBLAS)|
Llama-3-70B|fp16|2.85|75.33|137.8|-|
Llama-3-70B|w4g128|3.17|74.57|38.9|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w4g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-3-70b-EfficientQAT-w4g128-BitBLAS)|
Llama-3-70B|w3g128|4.19|72.42|32.2|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w3g128)|
Llama-3-70B|w2g64|6.08|67.89|23.2|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w2g64)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w2g64-GPTQ)|
Llama-3-70B|w2g128|6.38|67.57|22.0|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w2g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w2g128-GPTQ)\|[BitBLAS](Llama-3-70b-EfficientQAT-w2g128-BitBLAS)|
Llama-3-8B-Instruct|fp16|8.29|68.43|13.0|-|
Llama-3-8B-Instruct|w4g128|7.93|68.39|5.4|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w4g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-3-8b-instruct-EfficientQAT-w4g128-BitBLAS)|
Llama-3-8B-Instruct|w3g128|8.55|67.24|4.7|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w3g128)|
Llama-3-8B-Instruct|w2g64|11.19|60.66|3.9|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g64)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g64-GPTQ)\|[BitBLAS](Llama-3-8b-instruct-EfficientQAT-w2g64-BitBLAS)|
Llama-3-8B-Instruct|w2g128|11.73|60.16|3.8|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ)\|[BitBLAS](Llama-3-8b-instruct-EfficientQAT-w2g128-BitBLAS)|
Llama-3-70B-Instruct|fp16|5.33|73.78|137.8|-|
Llama-3-70B-Instruct|w4g128|5.35|73.47|38.9|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w4g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-3-70b-instruct-EfficientQAT-w4g128-BitBLAS)|
Llama-3-70B-Instruct|w3g128|5.65|72.87|32.2|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w3g128)|
Llama-3-70B-Instruct|w2g64|7.86|67.64|23.2|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w2g64)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w2g64-GPTQ)\|[BitBLAS](Llama-3-70b-instruct-EfficientQAT-w2g64-BitBLAS)|
Llama-3-70B-Instruct|w2g128|8.14|67.54|22.0|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w2g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w2g128-GPTQ)\|[BitBLAS](Llama-3-70b-instruct-EfficientQAT-w2g128-BitBLAS)|

## Usage of EQAT models
Please refer [https://github.com/OpenGVLab/EfficientQAT](https://github.com/OpenGVLab/EfficientQAT?tab=readme-ov-file#inference) for details.

## Usage of GPTQ and BitBLAS models
Below is an example to inference with GPTQ or BitBLAS quantized formats.
```Python
from transformers import AutoTokenizer
from gptqmodel import GPTQModel

quant_dir = "ChenMnZ/Llama-2-7b-EfficientQAT-w2g128-GPTQ"
# quant_dir = "ChenMnZ/Llama-2-7b-EfficientQAT-w2g128-BitBLAS"
# or local path

tokenizer = AutoTokenizer.from_pretrained(quant_dir, use_fast=True)


# load quantized model to the first GPU
model = GPTQModel.from_quantized(quant_dir)

# inference with model.generate
print(tokenizer.decode(model.generate(**tokenizer("Model quantization is", return_tensors="pt").to(model.device))[0]))
```


## Citation
If you found this work useful, please consider citing:
```
@article{efficientqat,
  title={EfficientQAT: Efficient Quantization-Aware Training for Large Language Models},
  author={Chen, Mengzhao and Shao, Wenqi and Xu, Peng and Wang, Jiahao and Gao, Peng and Zhang, Kaipeng and Qiao, Yu and Luo, Ping},
  journal={arXiv preprint arXiv:2407.11062},
  year={2024}
}
```