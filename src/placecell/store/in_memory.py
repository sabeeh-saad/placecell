"""In-process reference store with indexed metadata and bounded exact vector scans.

Its database lives on a single connection, so reads wait for a running transaction.
"""

from placecell.store.base import EVERYTHING
from placecell.store.state import StateStore


class InMemoryStore(StateStore):
    def close(self) -> None:
        """Clear the reference store; it remains usable as an empty store."""
        with self.transaction():
            self.delete_where(EVERYTHING)
            self.objects.clear()
