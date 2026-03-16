from logprobs import LogProbSampler

sampler = LogProbSampler()

# 1. Generate tokens and get logprobs at each position
print("=== Generation logprobs ===")
result = sampler.sample("The quick brown fox", max_tokens=5, top_logprobs=10)
print(f"Generated: {result.generated_text}")
for pos in result.tokens:
    print(f"\n  Chosen: {pos.chosen.token!r} (id={pos.chosen.token_id}, logprob={pos.chosen.logprob:.4f})")
    for t in pos.top_k:
        print(f"    {t.token!r:>15} id={t.token_id:<8} logprob={t.logprob:.4f}")

# 2. Get logprobs over the prompt tokens themselves
print("\n=== Prompt token logprobs ===")
result = sampler.prompt_logprobs("The quick brown fox jumps over the lazy dog", top_logprobs=5)
for pos in result.tokens:
    print(f"  {pos.chosen.token!r:>15} logprob={pos.chosen.logprob:.4f}")

# 3. Batch multiple prompts
print("\n=== Batch sampling ===")
results = sampler.batch_sample(
    ["Hello world", "Once upon a time", "import numpy as np"],
    max_tokens=3,
    top_logprobs=5,
)
for r in results:
    print(f"  {r.prompt!r:>30} -> {r.generated_text!r}")
