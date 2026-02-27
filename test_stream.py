import requests

response = requests.post(
    "http://localhost:8000/ask/stream",
    json={"question": "How do I create a FastAPI route?"},
    stream=True
)

# Use iter_lines for better streaming behavior
for line in response.iter_lines(decode_unicode=True):
    if line:
        print(line, flush=True)

print()  # Final newline
