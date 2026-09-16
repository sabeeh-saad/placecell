"""Wall-clock limits for the simulation's manual velocity input."""

import math
import time


class VelocityWatchdog:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.velocity, self.received_at = (0.0, 0.0), None

    def update(self, linear, angular):
        self.velocity = (
            (max(-0.35, min(0.35, linear)), max(-0.8, min(0.8, angular)))
            if math.isfinite(linear) and math.isfinite(angular)
            else (0.0, 0.0)
        )
        self.received_at = self.clock()

    def command(self):
        if self.received_at is None or not 0 <= self.clock() - self.received_at < 0.5:
            return 0.0, 0.0
        return self.velocity
