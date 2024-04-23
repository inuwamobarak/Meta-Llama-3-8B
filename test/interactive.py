import requests
import sys

sys.path.insert(0, '..')
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'


class InteractivePirateChatbot:
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
                                                           offload_buffers=True)

    def _prepare_inputs(self, messages):
        try:
            inputs = self._tokenizer([message['content'] for message in messages], padding='longest', truncation=True,
                                     max_length=512, return_tensors='pt')
            input_ids = inputs.input_ids.to(self._model.device)
            attention_mask = inputs.attention_mask.to(self._model.device)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        except Exception as e:
            print(f"Error preparing inputs: {e}")
            return None

    def ask(self, question):
        try:
            messages = [
                {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                {"role": "user", "content": question}
            ]

            prepared_data = self._prepare_inputs(messages)
            if prepared_data is None:
                print("Error preparing inputs. Skipping...")
                return

            output = self._model.generate(**prepared_data, max_length=512, num_beams=5, early_stopping=True)

            answer = self._tokenizer.decode(output[0], skip_special_tokens=True)
            print("Pirate:", answer)
        except Exception as e:
            print(f"Error generating response: {e}")


generator = InteractivePirateChatbot()
while True:
    question = input("User: ")
    generator.ask(question)
