import json
import re

INPUT_FILE = "nano_docs.jsonl"
OUTPUT_FILE = "nano_docs_clean.jsonl"

def fix_message(msg):
    """
    Attempt to fix a single message into { "role": "...", "content": "..." }.
    Possible cases:
      1. Already valid (dict with "role" and "content").
      2. A string like "system: You are a Nano expert." -> parse role, content
      3. A list with [role, content].
      4. Something else -> fallback to "unknown" role & stringified msg as content.
    """
    # CASE 1: Already valid dict
    if isinstance(msg, dict):
        # ensure mandatory keys exist
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        return {"role": str(role), "content": str(content)}

    # CASE 2: A string that might look like "system: foo bar"
    if isinstance(msg, str):
        # Attempt to parse role & content by looking for the first colon
        # e.g., "system: You are a Nano expert." -> role=system, content=You are...
        match = re.match(r"^([^:]+):(.*)$", msg.strip())
        if match:
            role = match.group(1).strip()
            content = match.group(2).strip()
            return {"role": role, "content": content}
        else:
            # Fallback: entire string is content, role=unknown
            return {"role": "unknown", "content": msg}

    # CASE 3: A list that might look like ["system", "You are a Nano expert."]
    if isinstance(msg, list):
        if len(msg) == 2:
            # interpret first item as role, second as content
            role = str(msg[0])
            content = str(msg[1])
            return {"role": role, "content": content}
        else:
            # fallback: put entire list as content
            return {"role": "unknown", "content": str(msg)}

    # CASE 4: Fallback
    return {"role": "unknown", "content": str(msg)}

def fix_record(record):
    """
    Attempt to fix one record so record["messages"] is a list of {role, content}.
    """
    if not isinstance(record, dict):
        return None

    messages = record.get("messages", None)
    if not isinstance(messages, list):
        # If it's not a list, attempt to parse or fallback
        # Possibly messages is a string, or something else
        # We'll convert it to a list with one item
        if messages:
            messages = [messages]
        else:
            messages = []

    fixed_messages = []
    for msg in messages:
        fixed_msg = fix_message(msg)
        fixed_messages.append(fixed_msg)

    record["messages"] = fixed_messages
    return record

def main():
    valid_count = 0
    fixed_count = 0
    invalid_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping line {i} - invalid JSON")
                invalid_count += 1
                continue

            # Attempt to fix
            fixed = fix_record(record)
            if fixed is None:
                print(f"Skipping line {i} - unable to fix record structure.")
                invalid_count += 1
                continue

            # We consider it valid now
            outfile.write(json.dumps(fixed) + "\n")
            valid_count += 1

    print(f"✅ Completed cleaning & fixing.")
    print(f"Valid (fixed) records: {valid_count}")
    print(f"Invalid (skipped) records: {invalid_count}")

if __name__ == "__main__":
    main()
