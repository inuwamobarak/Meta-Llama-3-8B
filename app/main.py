from flask import Flask, request, jsonify
import transformers
import torch

app = Flask(__name__)

# Initialize the model and pipeline outside of the function to avoid unnecessary reloading
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({'error': 'No message provided.'}), 400

    # Create system message
    messages = [{"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"}]

    # Add user message
    messages.append({"role": "user", "content": user_message})

    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    generated_text = outputs[0]['generated_text'][len(prompt):].strip()
    response = {
        'message': generated_text
    }

    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True)