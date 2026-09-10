"""placecell: long-term visual memory for mobile robots."""

from placecell.lifecycle import Curator, CuratorReport, ReinforcementPolicy, Reinforcer, RetentionPolicy
from placecell.memory import SCHEMA_VERSION, Evidence, EvidenceKind, Memory, Pose, memory_id
from placecell.pipeline import Ingester, IngestReport, Observation, SegmentationPolicy, Segmenter
from placecell.retrieval import RankedMemory, Recall
from placecell.store import CollectionInfo, Filter, Hit, InMemoryStore, VectorStore

__version__ = "0.0.1"

__all__ = [
    "SCHEMA_VERSION",
    "CollectionInfo",
    "Curator",
    "CuratorReport",
    "Evidence",
    "EvidenceKind",
    "Filter",
    "Hit",
    "InMemoryStore",
    "IngestReport",
    "Ingester",
    "Memory",
    "Observation",
    "Pose",
    "RankedMemory",
    "Recall",
    "ReinforcementPolicy",
    "Reinforcer",
    "RetentionPolicy",
    "SegmentationPolicy",
    "Segmenter",
    "VectorStore",
    "__version__",
    "memory_id",
]
