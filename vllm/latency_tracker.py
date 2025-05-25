from collections import defaultdict
from vllm.logger import init_logger

logger = init_logger(__name__)

# Concerto utils
class LatencyTracker:
    '''Latencies for a sequence group. Containing TimeToFristToken and TimePerOutputToken latencies.

    Args:
        request_id: The ID of the request.
        time_to_first_token: The time to first token latency.
        time_per_output_token: The time per output token latency.
    '''

    def __init__(self) -> None:
        self.arrival_timestamp = {}
        self.token_timestamps = defaultdict(list)
        self.base_ts: float = None

    def add_arrival_time(self, request_id: str, arrival_time: float) -> None:
        self.arrival_timestamp[request_id] = arrival_time

    def add_token_time(self, request_id: str, token_time: float) -> None:
        self.token_timestamps[request_id].append(token_time)

    def get_ttft(self, request_id: str) -> tuple[float, float]:
        if request_id not in self.token_timestamps:
            logger.warning(f"Request {request_id} does not have timestamps.")
            return float('nan'), float('nan')
        ttft = (self.token_timestamps[request_id][0] -
                self.arrival_timestamp[request_id]) * 1e3  # Return in ms.
        ts = (self.token_timestamps[request_id][0] - self.base_ts) * 1e3
        return ttft, ts

    def get_itl(self, request_id: str) -> tuple[list[float], list[float]]:
        if request_id not in self.token_timestamps:
            logger.warning(f"Request {request_id} does not have timestamps.")
            return [], []
        itl = []
        ts = []
        token_ts = self.token_timestamps[request_id]
        for i in range(1, len(token_ts)):
            itl.append((token_ts[i] - token_ts[i - 1]) * 1e3)  # in ms.
            ts.append((token_ts[i] - self.base_ts) * 1e3)  # in ms.
        return itl, ts

    def get_ft_ts(self, request_id: str) -> float:
        """ Get first token timestamp (for offline tput calculation). """
        if request_id not in self.token_timestamps:
            logger.warning(f"Request {request_id} does not have timestamps.")
            return 0.0
        token_ts = self.token_timestamps[request_id]
        ft_ts = (token_ts[0] - self.base_ts) * 1e3  # in ms.
        return ft_ts

    def get_pot_ts(self, request_id: str) -> list[float]:
        """ Get per-output token timestamp (for offline tput calculation). """
        if request_id not in self.token_timestamps:
            logger.warning(f"Request {request_id} does not have timestamps.")
            return 0.0
        pot_ts = []
        token_ts = self.token_timestamps[request_id]
        for i in range(1, len(token_ts)):
            pot_ts.append((token_ts[i] - self.base_ts) * 1e3)  # in ms.
        return pot_ts

    def untrack_req(self, request_id: str) -> None:
        self.arrival_timestamp.pop(request_id, None)
        self.token_timestamps.pop(request_id, None)
