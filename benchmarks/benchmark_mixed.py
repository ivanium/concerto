"""Benchmark mixed online/offline inference throughput. Online requests follow
the gamma distribution, and offline requests are given ahead of time.
"""
import argparse
import dataclasses
import random
import time
from typing import Optional
import numpy as np
import os
from collections import namedtuple

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.inputs import TokensPrompt

os.environ["VLLM_USE_V1"] = "0"

BenchReq = namedtuple("BenchReq", ["prompt_ids", "input_len", "output_len"])


def dummy_tokenizer(input_len: int):
    # [1] is the <s> token. and [7251] is the token for "hi"
    prompt_ids = [1] + [7251] * (input_len - 1)
    return prompt_ids


def gen_gamma_timeseries(alpha: float, beta: float, num_samples: int):
    """
    Generate a series of timestamps modeled by a gamma distribution.
    """
    # Gamma distributed intervals, shape = alpha, scale = 1/beta
    intervals = np.random.gamma(alpha, 1 / beta, num_samples)
    # Convert intervals to timestamps
    timestamps = np.cumsum(intervals)
    return timestamps


def sample_requests(
    dataset_path: Optional[str],
    num_requests: int,
    fixed_output_len: Optional[int],
) -> list[BenchReq]:
    dataset = np.loadtxt(dataset_path, dtype=int).reshape(-1, 2)
    indices = list(range(dataset.shape[0]))
    sampled_indices = random.sample(indices, num_requests)

    sampled_requests = []
    for idx in sampled_indices:
        input_len, output_len = dataset[idx]
        if fixed_output_len is not None:
            output_len = fixed_output_len
        prompt_ids = dummy_tokenizer(input_len)
        sampled_requests.append(BenchReq(prompt_ids, input_len, output_len))
    return sampled_requests


def load_online_dataset(
    dataset_path: Optional[str],
    num_requests: int,
    capped_input_len: Optional[int],
    capped_output_len: Optional[int],
) -> tuple[list[float], list[BenchReq]]:
    dataset = np.loadtxt(dataset_path, dtype=int, delimiter=",").reshape(-1, 3)
    timestamps = dataset.T[0].tolist()
    input_lens = dataset.T[1]
    output_lens = dataset.T[2]

    timestamps = timestamps[:num_requests]
    reqs = []
    for i in range(len(timestamps)):
        input_len, output_len = input_lens[i], output_lens[i]
        if capped_input_len is not None:
            input_len = min(capped_input_len, input_len)
        if capped_output_len is not None:
            output_len = min(capped_output_len, output_len)

        prompt_ids = dummy_tokenizer(input_len)
        reqs.append(BenchReq(prompt_ids, input_len, output_len))
    print(f"Loaded {len(reqs)} online requests from dataset {dataset_path}")
    return timestamps, reqs


def load_offline_dataset(
    dataset_path: Optional[str],
    num_requests: int,
    capped_input_len: Optional[int],
    capped_output_len: Optional[int],
) -> list[BenchReq]:
    dataset = np.loadtxt(dataset_path, dtype=int, delimiter=",").reshape(-1, 2)
    dataset = dataset[:num_requests]
    input_lens = dataset.T[0]
    output_lens = dataset.T[1]

    reqs = []
    for i in range(num_requests):
        idx = i % len(input_lens)
        input_len, output_len = input_lens[idx], output_lens[idx]
        if capped_input_len is not None:
            input_len = min(capped_input_len, input_len)
        if capped_output_len is not None:
            output_len = min(capped_output_len, output_len)

        prompt_ids = dummy_tokenizer(input_len)
        reqs.append(BenchReq(prompt_ids, input_len, output_len))
    print(f"Loaded {len(reqs)} offline requests from dataset {dataset_path}")
    return reqs


def gen_mixed_requests(
    dataset_path: Optional[str],
    input_len: Optional[int],
    output_len: Optional[int],
    offline_dataset_path: Optional[str],
    offline_input_len: Optional[int],
    offline_output_len: Optional[int],
    alpha: float,
    beta: float,
    num_online_reqs: int,
    num_offline_reqs: Optional[int],
) -> tuple[list[float], list[BenchReq], list[BenchReq]]:
    print("alpha: " + str(alpha))
    print("beta: " + str(beta))
    if num_offline_reqs is None:
        num_offline_reqs = 5 * num_online_reqs  # sufficient offline requests.

    # Sample the requests.
    if dataset_path is None:
        assert input_len is not None and output_len is not None
        # Synthesize a prompt with the given input length.
        prompt_ids = dummy_tokenizer(input_len)
        online_reqs = [
            BenchReq(prompt_ids, input_len, output_len)
            for _ in range(num_online_reqs)
        ]
        # Generate the timestamps for the online requests.
        online_timestamps = gen_gamma_timeseries(alpha, beta, num_online_reqs)
    else:
        online_timestamps, online_reqs = load_online_dataset(
            dataset_path, num_online_reqs, input_len, output_len)

    if offline_dataset_path is None:
        if offline_input_len is None:
            offline_input_len = input_len
        if offline_output_len is None:
            offline_output_len = output_len

        assert offline_input_len is not None and offline_output_len is not None
        # Synthesize a prompt with the given input length.
        prompt_ids = dummy_tokenizer(offline_input_len)
        offline_reqs = [
            BenchReq(prompt_ids, offline_input_len, offline_output_len)
            for _ in range(num_offline_reqs)
        ]
    else:
        offline_reqs = load_offline_dataset(offline_dataset_path,
                                            num_offline_reqs,
                                            offline_input_len,
                                            offline_output_len)

    return online_timestamps, online_reqs, offline_reqs


def warmup(llm: LLM, reqs: list[BenchReq]):
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=256,
    )
    num_reqs = 100
    reqs = reqs[:num_reqs]
    token_prompts = [TokensPrompt(prompt_token_ids=r.prompt_ids) for r in reqs]

    print("Warmup the model...")
    _ = llm.generate(prompts=token_prompts, sampling_params=sampling_params)
    print("Warmup done.")


def run_concerto(
    online_timestamps: list[float],
    online_requests: list[BenchReq],
    offline_requests: list[BenchReq],
    n: int,
    engine_args: EngineArgs,
    disable_detokenize: bool = False,
    batching_with_offline_reqs: bool = True,
    enable_profile: bool = False,
    enable_naive_colocation: bool = False,
    enable_naive_preemption: bool = False,
    enable_checkpointing: bool = False,
    enable_prefetching: bool = False,
    ttft_sla: Optional[int] = None,
    itl_sla: Optional[int] = None,
    detailed_stats_logging: bool = False,
) -> float:

    llm = LLM(**dataclasses.asdict(engine_args))

    warmup(llm, online_requests)

    if batching_with_offline_reqs:
        # Add offline requests to the engine.
        offline_prompts = [
            TokensPrompt(prompt_token_ids=req.prompt_ids)
            for req in offline_requests
        ]
        offline_sampling_params = [
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=req.output_len,
                detokenize=not disable_detokenize,
            ) for req in offline_requests
        ]
        llm.add_offline_dataset(prompts=offline_prompts,
                                sampling_params=offline_sampling_params)

    # Add online requests to the engine.
    assert len(online_timestamps) == len(online_requests)
    online_prompts = [
        TokensPrompt(prompt_token_ids=req.prompt_ids)
        for req in online_requests
    ]
    online_sampling_params = [
        SamplingParams(
            n=n,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=req.output_len,
            detokenize=not disable_detokenize,
        ) for req in online_requests
    ]

    start = time.perf_counter()
    (online_outputs, offline_outputs, raw_online_ttfts, raw_online_itls,
     raw_offline_ft_ts, raw_offline_pot_ts) = llm.run_online_trace(
         timestamps=online_timestamps,
         prompts=online_prompts,
         sampling_params=online_sampling_params,
         use_tqdm=True,
     )
    end = time.perf_counter()

    # Collect online stats
    ttfts = [lat for _, lat, ts in raw_online_ttfts]
    avg_itls = [sum(lats) / len(lats) for _, lats, ts in raw_online_itls]
    p99_ttft = np.percentile(ttfts, 99)
    p99_avg_itl = np.percentile(avg_itls, 99)
    p999_ttft = np.percentile(ttfts, 99.9)
    p999_avg_itl = np.percentile(avg_itls, 99.9)
    print(f"Average TTFT: {sum(ttfts) / len(ttfts):.2f} ms")
    print(f"Average ITL: {sum(avg_itls) / len(avg_itls):.2f} ms")
    print()
    print(f"P99 TTFT: {p99_ttft:.2f} ms")
    print(f"P99 ITL: {p99_avg_itl:.2f} ms")
    print(f"P999 TTFT: {p999_ttft:.2f} ms")
    print(f"P999 ITL: {p999_avg_itl:.2f} ms")
    print()
    num_online_reqs = len(online_outputs)
    num_online_tokens = sum(
        len(stp.token_ids) for output in online_outputs for stp in output.outputs)
    online_rps = num_online_reqs / (end - start)
    online_tps = num_online_tokens / (end - start)
    print(f"Online Tput: {online_rps:.2f} reqs/s, {online_tps:.2f} tokens/s")

    # Collect offline stats
    num_offline_reqs = len(offline_outputs)
    num_offline_tokens = sum(
        len(stp.token_ids) for output in offline_outputs for stp in output.outputs)
    offline_rps = num_offline_reqs / (end - start)
    offline_tps = num_offline_tokens / (end - start)
    print(
        f"Offline Tput: {offline_rps:.2f} reqs/s, {offline_tps:.2f} tokens/s")
    print(f"Finished offline requests: {num_offline_reqs}")

    # Dump detailed stats
    if detailed_stats_logging:
        with open("log/overall.txt", "w") as f:
            f.write(f"Average TTFT: {sum(ttfts) / len(ttfts):.2f} ms\n")
            f.write(f"Average ITL: {sum(avg_itls) / len(avg_itls):.2f} ms\n")
            f.write(f"P99 TTFT: {p99_ttft:.2f} ms\n")
            f.write(f"P99 ITL: {p99_avg_itl:.2f} ms\n")
            f.write(
                f"Online Tput: {len(online_outputs) / (end - start):.2f} reqs/s\n"
            )
            f.write(f"Offline Tput: {offline_rps:.2f} reqs/s\n")
            f.write(f"Offline Tput: {offline_tps:.2f} tokens/s\n")
            f.write(f"Finished offline requests: {num_offline_reqs}\n")
        # For performance debug
        with open("log/online_ttft.txt", "w") as f:
            f.write("Request_id, ttft\n")
            for rid, lat, _ in raw_online_ttfts:
                f.write(f"{rid}, {lat:.2f}\n")

        with open("log/online_itl.txt", "w") as f:
            f.write("Request_id, avg_itl, itl\n")
            for rid, lats, _ in raw_online_itls:
                avg_itl = sum(lats) / len(lats)
                f.write(f"{rid}, {avg_itl:.2f}, {lats}\n")

        # Online detailed stats
        raw_online_ttfts.sort(key=lambda x: x[2])
        with open("log/online_ttft_ts.txt", "w") as f:
            for rid, lat, ts in raw_online_ttfts:
                f.write(f"{ts:.2f} {lat:.2f} {rid}\n")

        online_itls = [(rid, itl, t) for rid, itls, ts in raw_online_itls
                       for itl, t in zip(itls, ts)]
        online_itls = sorted(online_itls, key=lambda x: x[2])
        with open("log/online_itl_ts.txt", "w") as f:
            for rid, lat, ts in online_itls:
                f.write(f"{ts:.2f} {lat:.2f} {rid}\n")

        online_token_ts = []
        for i, (rid, _, ts) in enumerate(raw_online_ttfts):
            num_tokens = online_requests[i][1]
            online_token_ts.append((rid, ts, num_tokens))
        for rid, _, pot_ts in raw_online_itls:
            for t in pot_ts:
                online_token_ts.append((rid, t, 1))
        online_token_ts.sort(key=lambda x: x[1])
        with open("log/online_token_ts.txt", "w") as f:
            for rid, ft_ts, num_tokens in online_token_ts:
                f.write(f"{ft_ts:.2f} {num_tokens} {rid}\n")

        # Offline detailed stats
        offline_token_ts = raw_offline_ft_ts
        for req_id, pot_ts in raw_offline_pot_ts:
            for t in pot_ts:  # 1 decode token per iteration
                offline_token_ts.append((req_id, t, 1))
        offline_token_ts.sort(key=lambda x: x[1])
        with open("log/offline_token_ts.txt", "w") as f:
            for rid, ft_ts, num_tokens in offline_token_ts:
                f.write(f"{ft_ts:.2f} {num_tokens} {rid}\n")

    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Sample the requests.
    online_timestamps, online_reqs, offline_reqs = gen_mixed_requests(
        args.dataset, args.input_len, args.output_len, args.offline_dataset,
        args.offline_input_len, args.offline_output_len, args.alpha, args.beta,
        args.num_online_prompts, args.num_offline_prompts)

    run_concerto(
        online_timestamps,
        online_reqs,
        offline_reqs,
        args.n,
        EngineArgs.from_cli_args(args),
        detailed_stats_logging=args.detailed_stats_logging,
        batching_with_offline_reqs=not args.disable_offline_batching,
        enable_profile=args.enable_profile,
        enable_naive_colocation=args.enable_naive_colocation,
        enable_naive_preemption=args.enable_naive_preemption,
        enable_checkpointing=args.enable_checkpointing,
        enable_prefetching=args.enable_prefetching,
        ttft_sla=args.ttft_sla,
        itl_sla=args.itl_sla,
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark online-offline co-serving performance.")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--offline-dataset",
                        type=str,
                        default=None,
                        help="Path to the offline dataset.")
    parser.add_argument("--alpha",
                        type=float,
                        default=0.5,
                        help="Alpha for the gamma distribution.")
    parser.add_argument("--beta",
                        type=float,
                        default=2,
                        help="Beta for the gamma distribution.")
    parser.add_argument(
        "--cv",
        type=float,
        default=None,
        help="Burstness for the gamma distribution. cv = 1/sqrt(alpha)")
    parser.add_argument("--rate",
                        type=float,
                        default=None,
                        help="Request Rate")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--offline-input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each offline request")
    parser.add_argument(
        "--offline-output-len",
        type=int,
        default=None,
        help="Output length for each offline request. Overrides "
        "the output length from the dataset.")
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--num-online-prompts",
                        type=int,
                        default=1000,
                        help="Number of online prompts to process.")
    parser.add_argument("--num-offline-prompts",
                        type=int,
                        default=None,
                        help="Number of offline prompts to process.")
    parser.add_argument(
        "--disable-offline-batching",
        action="store_true",
        help="Disable offline batching and serve online rquests only.")
    parser.add_argument("--detailed-stats-logging",
                        action="store_true",
                        help="Logging detailed latency statistics to files.")
    parser.add_argument("--enable-profile",
                        action="store_true",
                        help="Enable profiling.")
    parser.add_argument("--enable-naive-colocation",
                        action="store_true",
                        help="Enable naive colocation.")
    parser.add_argument("--enable-naive-preemption",
                        action="store_true",
                        help="Enable naive preemption.")
    parser.add_argument("--enable-checkpointing",
                        action="store_true",
                        help="Enable checkpointing.")
    parser.add_argument("--enable-prefetching",
                        action="store_true",
                        help="Enable prefetching.")
    parser.add_argument("--ttft-sla",
                        type=int,
                        default=None,
                        help="TTFT SLA in ms.")
    parser.add_argument("--itl-sla",
                        type=int,
                        default=None,
                        help="ITL SLA in ms.")

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    if args.cv is not None:
        args.alpha = 1.0 / (args.cv * args.cv)
    if args.rate is not None:
        args.beta = args.rate * args.alpha
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    if args.enable_profile:
        assert args.ttft_sla is not None and args.itl_sla is not None
    if args.preemption_mode == "recompute":
        assert args.enable_checkpointing is False
        assert args.enable_prefetching is False
    assert not (args.enable_naive_colocation and args.enable_checkpointing)
    assert not args.enable_naive_preemption or args.enable_naive_colocation

    main(args)
