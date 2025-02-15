import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./nano-llm"  # Change this if needed
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load a sentence similarity model for evaluation
similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load test dataset
test_file = "test_data.jsonl"
with open(test_file, "r") as f:
    test_samples = [json.loads(line) for line in f]

def generate_response(question):
    """Generate model response for a given question."""
    input_ids = tokenizer(question, return_tensors="pt").input_ids
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=256)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def evaluate_model():
    """Run test samples and compare results."""
    correct = 0
    total = len(test_samples)

    for sample in test_samples:
        question = sample["question"]
        expected_answer = sample["expected_answer"]

        generated_answer = generate_response(question)

        # Compute similarity score
        similarity = util.pytorch_cos_sim(
            similarity_model.encode(generated_answer, convert_to_tensor=True),
            similarity_model.encode(expected_answer, convert_to_tensor=True)
        ).item()

        # Consider it correct if similarity > 0.75
        is_correct = similarity > 0.75
        correct += is_correct

        print(f"\n🔹 Question: {question}")
        print(f"✅ Expected: {expected_answer}")
        print(f"🤖 Model Output: {generated_answer}")
        print(f"📊 Similarity Score: {similarity:.2f} ({'✔️ Correct' if is_correct else '❌ Incorrect'})")

    accuracy = (correct / total) * 100
    print(f"\n🎯 Model Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

# Run evaluation
evaluate_model()
