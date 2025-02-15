import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./nano-llm"  # Change this if needed
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def chat():
    print("🤖 Nano LLM - Ask me anything about Nano! (Type 'exit' to quit)")

    while True:
        user_input = input("📝 You: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        input_ids = tokenizer(user_input, return_tensors="pt").input_ids
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=256)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print(f"🤖 NanoBot: {response}\n")

# Run chatbot
if __name__ == "__main__":
    chat()
