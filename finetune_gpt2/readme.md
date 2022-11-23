## Finetuning GPT2

1. Make use of the existing finetuning script present from [huggingface](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py). 
2. Run with the desired datasets, ensuring to save to the root directory.
3. set flags ```--num_train_epochs 10``` and ```--save_strategy epoch```