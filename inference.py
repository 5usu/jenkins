from llama_cpp import Llama

# Load the quantized model
llm = Llama(model_path="./ggml_model/ggml-model-q8_0.bin")

# Generate a response
response = llm("How do I install Jenkins?")
print(response['choices'][0]['text'])
