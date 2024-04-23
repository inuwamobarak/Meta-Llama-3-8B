import requests
import sys

sys.path.insert(0, '../..')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

API_BASE_URL = 'http://localhost:5000/'
MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
TOKENIZER_PATH = MODEL_NAME + '/tokenizer_config.json'
MODEL_PATH = MODEL_NAME + '/checkpoint-best-bleu'


class InteractivePromptGenerator:
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).half().cuda()

    def _prepare_inputs(self, messages):
        inputs = self._tokenizer([message['content'] for message in messages], padding='longest', truncation=True,
                                 return_tensors='pt').input_ids
        attention_mask = inputs != self._tokenizer.pad_token_id
        return {'inputs': inputs.cuda(), 'attention_mask': attention_mask}

    def ask(self, question):
        messages = [{'role': 'assistant', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': question}]

        prepared_data = self._prepare_inputs(messages)
        output = self._model.generate(**prepared_data, max_length=512, num_beams=5, early_stopping=True)

        answer = self._tokenizer.decode(output[0])
        print("Assistant:", answer)


generator = InteractivePromptGenerator()
while True:
    question = input("User: ")
    generator.ask(question)