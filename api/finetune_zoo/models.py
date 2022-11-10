from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

ft_models = {
    '0': pipeline("text-generation", model="mrm8488/bloom-560m-finetuned-common_gen", device=device),
    '1': pipeline("text-generation", model="KoboldAI/OPT-350M-Nerys-v2", device=device),
    '2': pipeline("text-generation", model="LACAI/DialoGPT-large-PFG", device=device),
    '3': pipeline("text-generation", model="arminmehrabian/distilgpt2-finetuned-wikitext2-agu", device=device),
    '4': pipeline("text-generation", model="ethzanalytics/ai-msgbot-gpt2-XL", device=device),
    '5': pipeline("text-generation", model='dbmdz/german-gpt2', device=device),
    '6': pipeline("text-generation", model='wvangils/GPT-Neo-125m-Beatles-Lyrics-finetuned-newlyrics', device=device),
    '7': pipeline("text-generation", model='textattack/xlnet-base-cased-imdb', device=device),
    '8': pipeline("text-generation", model='veddm/paraphrase-multilingual-MiniLM-L12-v2-finetuned-DIT-10_epochs', device=device),
    '9': pipeline("text-generation", model="giulio98/CodeGen-350M-mono-xlcost", device=device),
}

'''

ft_train_models = {
    '10': pipeline("text-generation", model="wvangils/BLOOM-350m-Beatles-Lyrics-finetuned-newlyrics", device=device),
    '11': pipeline("text-generation", model="Tianyi98/opt-350m-finetuned-cola", device=device),
    '12': pipeline("text-generation", model="mdc1616/DialoGPT-large-sherlock", device=device),
    '13': pipeline("text-generation", model="noelmathewisaac/inspirational-quotes-distilgpt2", device=device),
    '14': pipeline("text-generation", model="malteos/gpt2-xl-wechsel-german", device=device),
    '15': pipeline("text-generation", model='lvwerra/gpt2-imdb', device=device),
    '16': pipeline("text-generation", model='flax-community/gpt-neo-125M-code-clippy', device=device),
    '17': pipeline("text-generation", model='textattack/xlnet-base-cased-rotten-tomatoes', device=device),
    '18': pipeline("text-generation", model='f00d/Multilingual-MiniLM-L12-H384-MLM-finetuned-wikipedia_bn_custom', device=device),
    '19': pipeline("text-generation", model="Salesforce/codegen-350M-mono", device=device),
}



'''