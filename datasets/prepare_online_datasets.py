import os
import argparse
from bisect import bisect_left, bisect_right
import numpy as np


def get_path(path: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def subrange(reqs, start, end):
    ts = [req[0] for req in reqs]
    stt_idx = bisect_left(ts, start)
    end_idx = bisect_right(ts, end)

    return reqs[stt_idx:end_idx]


def extract_day(reqs, stt_day, end_day):
    DAY_IN_S = 24 * 60 * 60
    stt = DAY_IN_S * stt_day
    end = DAY_IN_S * end_day
    return subrange(reqs, stt, end)


def extract_day_minutes(reqs, day, minute_stt, minute_end):
    DAY_IN_S = 24 * 60 * 60
    MINUTE_IN_S = 60
    stt = DAY_IN_S * day + MINUTE_IN_S * minute_stt
    end = DAY_IN_S * day + MINUTE_IN_S * minute_end
    return subrange(reqs, stt, end)


def load_dataset(path: str):
    data = np.loadtxt(path, dtype=int, delimiter=",")
    return data


def prepare_dataset(dataset_path: str,
                    output_path: str,
                    max_request_length: int,
                    dump: bool = True):
    output_path = f"{output_path}.txt"
    if os.path.exists(get_path(output_path)):
        print(f"Output file {output_path} already exists. Loading...")
        reqs = load_dataset(get_path(output_path))
        return reqs

    reqs = []
    with open(dataset_path) as f:
        lines = f.readlines()
        lines = lines[1:]  # Skip the head line.
        for line in lines:
            fields = line.split(",")
            if fields[1] != "ChatGPT":  # Only consider ChatGPT requests.
                continue

            timestamp, input_len, output_len = (
                int(fields[0]),
                int(fields[2]),
                int(fields[3]),
            )
            if input_len + output_len > max_request_length:
                continue  # Skip too long requests.

            reqs.append((timestamp, input_len, output_len))

        reqs = np.array(reqs, dtype=int)
        if dump:
            np.savetxt(get_path(output_path), reqs, fmt="%d,%d,%d")
        return reqs


def resample_dataset(reqs, day_stt: int, minute_stt: int, minute_end: int,
                     duration: int):
    reqs = extract_day_minutes(reqs, day_stt, minute_stt, minute_end)

    stt = reqs[0][0]
    end = reqs[-1][0]
    scale = (end - stt) / duration

    reqs[:, 0] -= stt
    reqs[:, 0] = (reqs[:, 0] / scale).astype(int)

    return reqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=get_path("burstgpt/BurstGPT_without_fails.csv"),
        help="Path to the BurstGPT dataset.",
    )
    parser.add_argument("--output-path",
                        type=str,
                        default="burstgpt/burstgpt-reqs",
                        help="Dataset name.")
    parser.add_argument("--max-request-length",
                        type=int,
                        default=None,
                        help="Maximum request length.")
    parser.add_argument("--day-stt",
                        type=int,
                        default=20,
                        help="Day to extract.")
    parser.add_argument("--minute-stt",
                        type=int,
                        default=1195,
                        help="Start minute to extract.")
    parser.add_argument("--minute-end",
                        type=int,
                        default=1210,
                        help="End minute to extract.")
    parser.add_argument("--duration",
                        type=int,
                        default=15,
                        help="Duration in minutes after resampling.")
    parser.add_argument(
        "--disable-dump",
        action="store_true",
        help="Disable dumping the extracted data to files.",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Enforce the resampling even if the output file exists.",
    )
    args = parser.parse_args()

    if args.max_request_length is None:
        args.max_request_length = 8192  # Filter out too long requests.

    print("Start preparing the dataset.")
    reqs = prepare_dataset(args.dataset_path, args.output_path,
                           args.max_request_length, not args.disable_dump)
    print("Dataset has been prepared.")
    print(f"Loaded {len(reqs)} requests in total.")

    day_stt = args.day_stt
    minute_stt, minute_end = args.minute_stt, args.minute_end
    duration_s = args.duration * 60  # to seconds
    output_path = f"{args.output_path}-sampled.txt"

    if os.path.exists(get_path(output_path)) and not args.enforce:
        print(
            f"Resampled data for day {day_stt} from minute {minute_stt} to {minute_end} already exists."
        )
    else:
        resampled = resample_dataset(reqs, day_stt, minute_stt, minute_end,
                                     duration_s)
        print(
            f"Resampled {len(resampled)} requests for day {day_stt} from minute {minute_stt} to {minute_end}."
        )
        np.savetxt(
            get_path(output_path),
            resampled,
            fmt="%d,%d,%d",
        )


main()
