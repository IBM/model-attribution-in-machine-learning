from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
import re

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))


tokenizer.pre_tokenizer = Whitespace()

base_model_names = ["bloom-350m", "DialoGPT-large", "distilgpt2", "gpt2", "Multilingual-MiniLM-L12-H384",
                    "gpt2-xl", "gpt-neo-125M", "opt-350m", "xlnet-base-cased", "codegen-350M-multi"]

def load_prompts() -> list:
    all_prompts = set()
    for root, dirs, files in os.walk(os.path.abspath(os.path.join(__file__, '../../prompts'))):
        for file in files:
            if re.search('[^\d].csv', file) and 'ppl' not in file and file.split('.')[0] not in base_model_names:
                with open(os.path.join(root, file), 'r') as prompt_file:
                    for line in prompt_file.readlines():
                        all_prompts.add(line)
                # dataset = os.path.join(root, file).split('/')[-2]
                # all_prompts[dataset] = list(prompts)
    return list(all_prompts)

prompts = load_prompts()

with open(os.path.abspath(os.path.join(__file__, '../../prompts/all_prompts.txt')), 'w') as f:
    for prompt in prompts:
        f.write(f"{prompt}\n")

tokenizer.train([os.path.abspath(os.path.join(__file__, '../../prompts/all_prompts.txt'))], trainer)
tokenizer.save('./bpe_tokenizer.json')
print(tokenizer.get_vocab_size())
