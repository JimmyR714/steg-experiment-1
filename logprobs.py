"""
Log probability sampling from Qwen2.5-14B-AWQ via vLLM (in-process, no server).

Usage:
    # CLI:
    python logprobs.py --prompt "The quick brown fox" --max-tokens 10 --top-logprobs 5

    # As a library:
    from logprobs import LogProbSampler
    sampler = LogProbSampler()
    result = sampler.sample("The quick brown fox", max_tokens=1, top_logprobs=5)
"""

import argparse
import json
from dataclasses import dataclass

from vllm import LLM, SamplingParams


DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"


@dataclass
class TokenLogProb:
    token: str
    token_id: int
    logprob: float


@dataclass
class PositionLogProbs:
    """Logprobs at a single token position."""
    chosen: TokenLogProb
    top_k: list[TokenLogProb]


@dataclass
class SampleResult:
    """Full result from a logprob sampling call."""
    prompt: str
    generated_text: str
    tokens: list[PositionLogProbs]

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "tokens": [
                {
                    "chosen": {"token": t.chosen.token, "token_id": t.chosen.token_id, "logprob": t.chosen.logprob},
                    "top_k": [{"token": tk.token, "token_id": tk.token_id, "logprob": tk.logprob} for tk in t.top_k],
                }
                for t in self.tokens
            ],
        }


class LogProbSampler:
    def __init__(self, model: str = DEFAULT_MODEL, max_model_len: int = 4096):
        self.llm = LLM(model=model, dtype="auto", max_model_len=max_model_len)

    def sample(
        self,
        prompt: str,
        max_tokens: int = 1,
        top_logprobs: int = 5,
        temperature: float = 0.0,
        prompt_logprobs: bool = False,
    ) -> SampleResult:
        """
        Sample tokens and return logprobs at each position.

        Args:
            prompt: Input text.
            max_tokens: Number of tokens to generate.
            top_logprobs: Number of top logprobs to return per position (max 20).
            temperature: Sampling temperature. 0.0 = greedy.
            prompt_logprobs: If True, also return logprobs for the prompt tokens.
        """
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=top_logprobs,
            prompt_logprobs=top_logprobs if prompt_logprobs else None,
        )

        outputs = self.llm.generate([prompt], params)
        output = outputs[0]

        tokens_out = []

        # Prompt token logprobs
        if prompt_logprobs and output.prompt_logprobs:
            prompt_token_ids = output.prompt_token_ids
            for i, lp_dict in enumerate(output.prompt_logprobs):
                if lp_dict is None:
                    # First token has no logprob
                    token_id = prompt_token_ids[i]
                    token_str = self.llm.get_tokenizer().decode([token_id])
                    chosen = TokenLogProb(token=token_str, token_id=token_id, logprob=0.0)
                    tokens_out.append(PositionLogProbs(chosen=chosen, top_k=[]))
                    continue

                token_id = prompt_token_ids[i]
                chosen_lp = lp_dict.get(token_id)
                token_str = chosen_lp.decoded_token if chosen_lp else self.llm.get_tokenizer().decode([token_id])
                chosen_logprob = chosen_lp.logprob if chosen_lp else 0.0
                chosen = TokenLogProb(token=token_str, token_id=token_id, logprob=chosen_logprob)

                top_k = [
                    TokenLogProb(token=lp.decoded_token, token_id=tid, logprob=lp.logprob)
                    for tid, lp in lp_dict.items()
                ]
                top_k.sort(key=lambda x: x.logprob, reverse=True)
                tokens_out.append(PositionLogProbs(chosen=chosen, top_k=top_k))

        # Generated token logprobs
        for step in output.outputs[0].logprobs:
            # step is a dict of {token_id: Logprob}
            # The chosen token is the one in the output
            chosen_id = max(step, key=lambda tid: step[tid].logprob)
            # Actually, the chosen token is the first key in the ordered dict
            for tid, lp in step.items():
                chosen_id = tid
                break
            chosen_lp = step[chosen_id]
            chosen = TokenLogProb(
                token=chosen_lp.decoded_token,
                token_id=chosen_id,
                logprob=chosen_lp.logprob,
            )

            top_k = [
                TokenLogProb(token=lp.decoded_token, token_id=tid, logprob=lp.logprob)
                for tid, lp in step.items()
            ]
            top_k.sort(key=lambda x: x.logprob, reverse=True)
            tokens_out.append(PositionLogProbs(chosen=chosen, top_k=top_k))

        return SampleResult(
            prompt=prompt,
            generated_text=output.outputs[0].text,
            tokens=tokens_out,
        )

    def prompt_logprobs(self, prompt: str, top_logprobs: int = 5) -> SampleResult:
        """Get logprobs for each token in the prompt itself (no generation)."""
        return self.sample(prompt, max_tokens=1, top_logprobs=top_logprobs, prompt_logprobs=True)

    def batch_sample(
        self,
        prompts: list[str],
        max_tokens: int = 1,
        top_logprobs: int = 5,
        temperature: float = 0.0,
    ) -> list[SampleResult]:
        """Sample logprobs for multiple prompts."""
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=top_logprobs,
        )
        outputs = self.llm.generate(prompts, params)
        results = []
        for output in outputs:
            tokens_out = []
            for step in output.outputs[0].logprobs:
                for tid, lp in step.items():
                    chosen_id = tid
                    break
                chosen_lp = step[chosen_id]
                chosen = TokenLogProb(
                    token=chosen_lp.decoded_token, token_id=chosen_id, logprob=chosen_lp.logprob
                )
                top_k = [
                    TokenLogProb(token=lp.decoded_token, token_id=tid, logprob=lp.logprob)
                    for tid, lp in step.items()
                ]
                top_k.sort(key=lambda x: x.logprob, reverse=True)
                tokens_out.append(PositionLogProbs(chosen=chosen, top_k=top_k))
            results.append(SampleResult(
                prompt=output.prompt, generated_text=output.outputs[0].text, tokens=tokens_out
            ))
        return results


def main():
    parser = argparse.ArgumentParser(description="Sample logprobs from Qwen2.5-14B-AWQ via vLLM")
    parser.add_argument("--prompt", required=True, help="Input prompt text")
    parser.add_argument("--max-tokens", type=int, default=10, help="Tokens to generate (default: 10)")
    parser.add_argument("--top-logprobs", type=int, default=5, help="Top-k logprobs per position (default: 5)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--prompt-logprobs", action="store_true", help="Include prompt token logprobs")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    sampler = LogProbSampler(model=args.model)
    result = sampler.sample(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        top_logprobs=args.top_logprobs,
        temperature=args.temperature,
        prompt_logprobs=args.prompt_logprobs,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Prompt: {result.prompt}")
        print(f"Generated: {result.generated_text}")
        print(f"\n{'Pos':<5} {'Token':<20} {'ID':<10} {'LogProb':<12} {'Top-k alternatives'}")
        print("-" * 90)
        for i, pos in enumerate(result.tokens):
            alt_str = ", ".join(f"{t.token!r}({t.logprob:.4f})" for t in pos.top_k[:3])
            print(f"{i:<5} {pos.chosen.token!r:<20} {pos.chosen.token_id:<10} {pos.chosen.logprob:<12.4f} {alt_str}")


if __name__ == "__main__":
    main()
