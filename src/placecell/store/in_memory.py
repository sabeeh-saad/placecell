"""In-process reference store with indexed metadata and bounded exact vector scans."""

from placecell.store.base import EVERYTHING
from placecell.store.state import StateStore


class InMemoryStore(StateStore):
    def close(self) -> None:
        """Clear the reference store; it remains usable as an empty store."""
        with self.transaction():
            self.delete_where(EVERYTHING)
            self.objects.clear()
