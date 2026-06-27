import torch
from transformers import MambaForCausalLM

# Load model
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf").to("cuda:0")

# Initial prompt tokens
input_ids = torch.load("input/input_ids.pt").to("cuda:0")

# First forward pass
with torch.no_grad():
    out = model(input_ids, use_cache=True)

cache = out.cache_params
logits = out.logits

# Get next token from final position
next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

# Continue generation using the cache
with torch.no_grad():
    out = model(
        input_ids=next_token,
        cache_params=cache,
        use_cache=True,
    )

# Updated cache and logits
cache = out.cache_params
logits = out.logits

dist_out = torch.load("results/logits_dist_4_2.pt").to("cuda:0")

orig_out = logits

# 1. allclose check
is_close = torch.allclose(
    dist_out,
    orig_out,
    rtol=1e-5,
    atol=1e-6
)

print("Allclose:", is_close)

# 2. RMSE
diff = dist_out - orig_out
rmse = torch.sqrt(torch.mean(diff ** 2))

print("RMSE:", rmse.item())

# 3. Maximum absolute difference
max_diff = torch.max(torch.abs(diff))

print("Max difference:", max_diff.item())

not_close = ~torch.isclose(
    dist_out,
    orig_out,
    rtol=1e-5,
    atol=1e-6
)

# Count mismatches
num_false = not_close.sum().item()

# Total number of elements
total = not_close.numel()

# Percentage
percent = 100 * num_false / total

print(f"Different elements: {num_false} / {total}")
print(f"Percentage different: {percent:.6f}%")