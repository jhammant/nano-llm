import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
MODEL_PATH = "meta-llama/Meta-Llama-3-8B"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Use MPS if available

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side='right')

tokenizer.pad_token = tokenizer.eos_token  # Ensure proper padding

SYSTEM_PROMPT = """
You are NanoBot, a cryptocurrency expert specializing in Nano. You provide **detailed, structured, and helpful answers** based on official Nano documentation.
**Rules:**
- Provide a **step-by-step** answer before linking to **https://docs.nano.org/**.
- **Avoid short responses** – be as **thorough and helpful as possible**.
- **Do NOT repeat instructions or generate follow-up questions**.
- **Only respond to the user’s exact query, nothing extra**.
"""

def chat():
    print("🤖 Nano LLM - Ask me anything about Nano! (Type 'exit' to quit)")

    conversation_history = SYSTEM_PROMPT + "\n"  # Maintain history for context

    while True:
        user_input = input("📝 You: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        conversation_history += f"\nUser: {user_input}\nNanoBot: "
        input_data = tokenizer(conversation_history, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        input_ids = input_data.input_ids
        attention_mask = input_data.attention_mask

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=2048,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.3,
                do_sample=True,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Extract only NanoBot's response, removing unintended prompts
        if "NanoBot:" in response:
            response = response.split("NanoBot:")[1].strip()

        print(f"🤖 NanoBot: {response}\n")

        # Append only the assistant’s response back to conversation history
        conversation_history += f"{response}\n"

# Run chatbot
if __name__ == "__main__":
    chat()
