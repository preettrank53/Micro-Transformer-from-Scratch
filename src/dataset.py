from pathlib import Path
import torch


# Resolve data path relative to this script's location
DATA_PATH = Path(__file__).parent.parent / "data" / "input.txt"

# Read the raw text dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Print total character count
print("Number of characters:", len(text))

# set(): unique characters | list(): convert to list | sorted(): sort alphabetically
chars = sorted(list(set(text)))

# Size of the vocabulary (total unique characters, V)
vocab_size = len(chars)

print("Vocabulary size:", vocab_size)
print("Vocabulary:")
print("".join(chars))


# Map: character -> integer (e.g., {'A': 0, 'B': 1, 'C': 2, 'D': 3})
char2int = {ch: i for i, ch in enumerate(chars)}

# Map: integer -> character (e.g., {0: 'A', 1: 'B', 2: 'C', 3: 'D'})
int2char = {i: ch for i, ch in enumerate(chars)}


# Convert a string of text into a list of integer IDs (e.g., "ABC" -> [0, 1, 2])
def encode(text):
    return [char2int[ch] for ch in text]


# Convert a list of integer IDs back into a string of text (e.g., [0, 1, 2] -> "ABC")
def decode(ids):
    return "".join(int2char[i] for i in ids)


# Test the tokenizer with a round-trip assertion
test_text = "Hello transformer!"
encoded = encode(test_text)
decoded = decode(encoded)

print("Original:", test_text)
print("Encoded:", encoded)
print("Decoded:", decoded)

assert decoded == test_text, "Tokenizer round-trip test failed!"


# Convert the entire dataset to a PyTorch long tensor for the embedding layer
data = torch.tensor(encode(text), dtype=torch.long)

print("Data shape:", data.shape)
print("Data type:", data.dtype)


# 90% train, 10% validation split to measure generalization on unseen text
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("Train data shape:", train_data.shape)
print("Val data shape:", val_data.shape)


# Batch parameters
batch_size = 8          # B: number of independent sequences to process in parallel
context_length = 128    # T: maximum context window size for predictions

# Generate a small batch of inputs (x) and targets (y)
def get_batch(split):
    # Select data source based on split
    data_source = train_data if split == "train" else val_data
    
    # Choose random starting index positions, avoiding overflow near the end
    ix = torch.randint(len(data_source) - context_length, (batch_size,))
    
    # Extract input context sequences (shape: B, T)
    x = torch.stack([data_source[i:i + context_length] for i in ix])
    
    # Extract target sequences shifted by 1 position (shape: B, T)
    y = torch.stack([data_source[i + 1:i + context_length + 1] for i in ix])
    
    return x, y

# Verify batch generation and tokenizer mapping
x, y = get_batch("train")
print("x shape:", x.shape)
print("y shape:", y.shape)
print("x:", x[0])
print("y:", y[0])
print("Decoded x:", decode(x[0].tolist()))
print("Decoded y:", decode(y[0].tolist()))

# Assertions to verify dimensions and the 1-position shift
assert x.shape == (batch_size, context_length)
assert y.shape == (batch_size, context_length)
assert torch.equal(x[:, 1:], y[:, :-1]), "Shift check failed!"



