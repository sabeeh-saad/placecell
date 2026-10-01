"""The reviewed mission a navigation controller is running: its plan, identity, step and grounding.

`NavigationCommands` asks the planner for a plan and then walks it one destination at a
time. `MissionSequencer` only holds that progress; the controller locks, resolves and
dispatches, and decides when the mission ends.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from placecell.missions import MissionPlan


class MissionSequencer:
    """Which reviewed plan runs, its current step and the user words that may name its places."""

    def __init__(self) -> None:
        self.plan: MissionPlan | None = None
        self.id = ""
        self.step = 0
        self.grounding: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return self.plan is not None

    def clear(self) -> None:
        self.plan, self.id, self.step = None, "", 0
        self.grounding = ()

    def start(self, plan: MissionPlan, text: str, context: Iterable[Mapping[str, Any]]) -> None:
        """Run a reviewed plan from its first step."""
        self.plan = plan
        # The user's words that a configured place must appear in: this request and the
        # earlier instructions the planner was shown.
        self.grounding = (
            text,
            *(
                event["data"]["text"]
                for event in context
                if event.get("kind") == "instruction" and isinstance(event.get("data", {}).get("text"), str)
            ),
        )

    def has_next(self) -> bool:
        return self.plan is not None and self.step + 1 < len(self.plan.destinations)

    def advance(self) -> str:
        """Move to the next step and return its destination."""
        assert self.plan is not None
        self.step += 1
        return self.plan.destinations[self.step]

    def status_fields(self) -> tuple[str, int, tuple[str, ...]]:
        """Mission ID, one-based step (0 once the plan is cleared) and destinations for a status."""
        return self.id, self.step + 1 if self.plan else 0, self.plan.destinations if self.plan else ()
