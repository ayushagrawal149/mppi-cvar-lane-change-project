"""Display / record helpers for the simulation drivers.

Three knobs the user can combine on top of `--render`:
- --real-time : pace the pygame window at wall-clock sim time (turns on
                HighwayEnv's `real_time_rendering` config).
- --slowdown F: extra wall-clock factor on top of real-time. F = 2 means
                the window plays at half speed; physics + controller are
                unchanged.
- --gif       : capture each frame and save sim_output/<run>/run.gif.
                Uses `render_mode="rgb_array"`, so a pygame window is NOT
                opened (the two modes are mutually exclusive in gym).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


class RenderHelper:
    def __init__(
        self,
        render: bool,
        gif: bool,
        slowdown: float = 1.0,
        real_time: bool = False,
        policy_dt: float = 0.05,
        gif_fps: int = 20,
        gif_name: str = "run.gif",
    ):
        if render and gif:
            raise ValueError(
                "--render and --gif are mutually exclusive; "
                "gym only supports one render_mode per env."
            )
        self.render = render
        self.gif = gif
        self.slowdown = max(float(slowdown), 1.0)
        self.real_time = bool(real_time) or self.slowdown > 1.0
        self.policy_dt = float(policy_dt)
        self.gif_fps = int(gif_fps)
        self.gif_name = gif_name
        self._frames: list = []

    @property
    def render_mode(self) -> str | None:
        if self.render:
            return "human"
        if self.gif:
            return "rgb_array"
        return None

    def configure_env(self, env) -> None:
        if self.render and self.real_time:
            env.unwrapped.config["real_time_rendering"] = True

    def after_step(self, env) -> None:
        if self.gif:
            frame = env.render()
            if frame is not None:
                self._frames.append(frame)
        if self.render and self.slowdown > 1.0:
            time.sleep((self.slowdown - 1.0) * self.policy_dt)

    def finalize(self, out_dir: Path) -> Path | None:
        if not self.gif or not self._frames:
            return None
        try:
            import imageio.v2 as iio
        except ImportError:
            import imageio as iio  # type: ignore[no-redef]
        path = Path(out_dir) / self.gif_name
        iio.mimsave(str(path), self._frames, fps=self.gif_fps, loop=0)
        return path

    def summary(self) -> dict[str, Any]:
        return {
            "render": self.render,
            "gif": self.gif,
            "slowdown": self.slowdown,
            "real_time": self.real_time,
            "frames_captured": len(self._frames),
        }
