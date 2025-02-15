import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
MODEL_PATH = "meta-llama/Meta-Llama-3-8B"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Use MPS if available

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side='right')

tokenizer.pad_token = tokenizer.eos_token  # Ensure proper padding

SYSTEM_PROMPT = """
You are a Nano cryptocurrency expert. Your job is to provide clear, detailed, and structured answers based on the official Nano documentation.
Always attempt to provide a step-by-step explanation before linking to https://docs.nano.org/.
If documentation is needed, append the relevant link **only at the end** of your response.
Ensure responses are detailed, avoid one-line answers, and focus on being helpful.
Do not generate follow-up questions unless explicitly asked to.
"""

def chat():
    print("🤖 Nano LLM - Ask me anything about Nano! (Type 'exit' to quit)")

    while True:
        user_input = input("📝 You: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}\nNanoBot:"  # Enforce system prompt
        input_data = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        input_ids = input_data.input_ids
        attention_mask = input_data.attention_mask

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=1024,  # Further increase length for detailed responses
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.5,  # Reduce randomness for more factual answers
                top_p=0.8,  # Prioritize likely responses
                repetition_penalty=1.2,  # Prevent repeated answers
                do_sample=True,  # Ensure more natural completions
                num_return_sequences=1,  # Ensure only one response is generated
            )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Ensure response is structured correctly
        response = response.strip().split("\nUser:")[0]  # Cut off any unintended follow-ups

        print(f"🤖 NanoBot: {response}\n")

# Run chatbot
if __name__ == "__main__":
    chat()
