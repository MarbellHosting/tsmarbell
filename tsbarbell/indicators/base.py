from typing import Any, Dict, List, Protocol


class ApiClient(Protocol):
    def _fetch(self, endpoint: str, limit: int = 10000) -> List[Dict[str, Any]]: ...


class BaseIndicator:
    def __init__(self, client: ApiClient):
        self.client = client
