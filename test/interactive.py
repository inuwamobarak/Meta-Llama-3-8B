import requests
import sys

sys.path.insert(0, '..')

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'


class InteractivePromptGenerator:
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).half().cuda()

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