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
- **Always include a relevant documentation link at the end**.
- **Keep responses concise and to the point. Avoid excessive explanations.**
- **Do NOT repeat instructions or generate follow-up questions**.
- **Only respond to the user’s exact query, nothing extra**.
- **Ensure responses remain factual and properly formatted**.
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
        input_data = tokenizer(conversation_history, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        input_ids = input_data.input_ids
        attention_mask = input_data.attention_mask

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=512,  # Reduce length to prevent unwanted format issues
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.5,  # Balanced randomness for structured responses
                top_p=0.8,
                repetition_penalty=1.5,  # Increase penalty to reduce repetitive/waffling responses
                do_sample=True,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Ensure response is structured correctly
        if "NanoBot:" in response:
            response = response.split("NanoBot:", 1)[-1].strip()

        # Remove unintended numbering artifacts
        response = "\n".join([line for line in response.split("\n") if not line.strip().isdigit()])

        # Trim excessive text beyond a reasonable length
        response = "\n".join(response.split("\n")[:10])  # Limit output to 10 meaningful lines

        # Append documentation link at the end if not already present
        if "https://docs.nano.org/" not in response:
            response += "\n\nFor more details, visit: https://docs.nano.org/running-a-node/"

        print(f"🤖 NanoBot: {response}\n")

        # Append only the assistant’s response back to conversation history
        conversation_history += f"{response}\n"

# Run chatbot
if __name__ == "__main__":
    chat()
