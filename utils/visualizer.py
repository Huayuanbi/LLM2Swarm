"""
utils/visualizer.py — Real-time swarm dashboard at 30 fps.

Architecture
────────────
  Main thread  : matplotlib FuncAnimation at 33 ms / frame (30 fps).
                 Only reads from _state_cache — no blocking calls.

  Poller thread: fetches SharedMemoryPool states every POLL_MS (100 ms)
                 via asyncio.run_coroutine_threadsafe and writes the result
                 into _state_cache under a lightweight threading.Lock.
                 Decouples the slow async fetch from the fast render loop.

Usage:
    viz = SwarmVisualizer(memory, drone_ids, loop)
    viz.show()   # blocks main thread; asyncio runs in a background thread
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Optional

import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from memory.pool import SharedMemoryPool

# ── Tunables ──────────────────────────────────────────────────────────────────
FPS          = 30
FRAME_MS     = 1000 // FPS    # 33 ms
POLL_MS      = 100            # state-fetch interval (ms) — independent of render
TRAIL_LEN    = 300            # positions kept per drone (10 s at 30 fps)
WORLD_RANGE  = 80             # axis half-extent in metres
ALT_MAX      = 30             # z-axis ceiling in metres

_STATUS_COLOUR = {
    "idle":       "#aaaaaa",
    "starting":   "#f0e040",
    "executing":  "#4caf50",
    "perceiving": "#2196f3",
    "replanning": "#ff9800",
    "hovering":   "#00bcd4",
    "error":      "#f44336",
    "landed":     "#795548",
}
_DEFAULT_COLOUR = "#ffffff"

_TRAIL_COLOURS = [
    "#64b5f6",  # drone_1 — sky blue
    "#ef9a9a",  # drone_2 — salmon
    "#a5d6a7",  # drone_3 — mint
    "#fff176",  # drone_4 — yellow
    "#ce93d8",  # drone_5 — lavender
]


class SwarmVisualizer:
    """
    30-fps real-time swarm dashboard.

    Args:
        memory:        Shared memory pool — read for status/task/observations (10 s cadence).
        drone_ids:     IDs to display.
        loop:          The asyncio event loop running in the swarm background thread.
        sim_positions: Simulator bridge dict ``{drone_id: [x,y,z]}`` written at 20 Hz
                       by the physics loop; decoupled from the memory pool.
        initial_plans: ``{drone_id: "takeoff(10m) → goto(…) → …"}`` populated by
                       run() after the GlobalOperator call; displayed in the plan panel.
        vlm_log:       Rolling deque of VLM decision strings written by drone lifecycles.
        task_queues:   ``{drone_id: ["▶ current", "next", ...]}`` updated each time an
                       action starts or finishes; shown in the live status panel.
    """

    def __init__(
        self,
        memory:        SharedMemoryPool,
        drone_ids:     list[str],
        loop:          asyncio.AbstractEventLoop,
        sim_positions: dict | None = None,
        initial_plans: dict | None = None,
        vlm_log:       deque | None = None,
        task_queues:   dict | None = None,
    ):
        self._memory        = memory
        self._drone_ids     = drone_ids
        self._loop          = loop
        self._stop_event    = threading.Event()
        self._sim_positions = sim_positions
        self._initial_plans = initial_plans   # {drone_id: summary string}
        self._vlm_log       = vlm_log         # shared rolling deque
        self._task_queues   = task_queues     # {drone_id: [label, ...]}

        # Shared cache written by poller, read by render — guarded by _cache_lock
        self._cache_lock  = threading.Lock()
        self._state_cache: dict = {}          # {drone_id: DroneState}

        # Per-drone position trail (deque for O(1) append/pop)
        self._trails: dict[str, deque] = {
            did: deque(maxlen=TRAIL_LEN) for did in drone_ids
        }

        # Status text is only updated every ~200 ms to avoid font-render cost
        self._text_cache:    str   = ""
        self._last_text_upd: float = 0.0
        TEXT_UPDATE_HZ = 5   # times per second
        self._text_interval  = 1.0 / TEXT_UPDATE_HZ

    # ── Public ─────────────────────────────────────────────────────────────────

    def show(self) -> None:
        """
        Start the poller, build the figure, run FuncAnimation.
        Blocks until the window is closed. Must be called from the main thread.
        """
        self._start_poller()
        self._build_figure()
        self._anim = FuncAnimation(
            self._fig,
            self._update,
            interval=FRAME_MS,
            cache_frame_data=False,
        )
        plt.show(block=True)

    def stop(self) -> None:
        self._stop_event.set()

    # ── Poller thread ──────────────────────────────────────────────────────────

    def _start_poller(self) -> None:
        t = threading.Thread(target=self._poll_loop, name="viz-poller", daemon=True)
        t.start()

    def _poll_loop(self) -> None:
        """Fetches memory pool every POLL_MS and updates _state_cache."""
        while not self._stop_event.is_set():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._memory.get_all_states(), self._loop
                )
                states = future.result(timeout=0.09)   # just under POLL_MS
                with self._cache_lock:
                    self._state_cache = states
            except Exception:
                # If fetch fails just keep the last known states
                pass
            time.sleep(POLL_MS / 1000)

    # ── Figure setup ───────────────────────────────────────────────────────────

    def _build_figure(self) -> None:
        plt.style.use("dark_background")
        self._fig = plt.figure(figsize=(16, 8), facecolor="#0d0d1a")
        self._fig.canvas.manager.set_window_title("LLM2Swarm — Live Dashboard")

        gs = gridspec.GridSpec(
            2, 2,
            width_ratios=[1.6, 1],
            height_ratios=[1, 1],
            wspace=0.04, hspace=0.08,
            left=0.02, right=0.98,
            top=0.93, bottom=0.04,
        )

        # ── 3D view (spans both rows of column 0) ──
        self._ax3d: Axes3D = self._fig.add_subplot(gs[:, 0], projection="3d")
        self._ax3d.set_facecolor("#0d0d1a")
        self._ax3d.set_xlim(-WORLD_RANGE, WORLD_RANGE)
        self._ax3d.set_ylim(-WORLD_RANGE, WORLD_RANGE)
        self._ax3d.set_zlim(0, ALT_MAX)
        self._ax3d.set_xlabel("X (m)", color="#555", labelpad=4)
        self._ax3d.set_ylabel("Y (m)", color="#555", labelpad=4)
        self._ax3d.set_zlabel("Z (m)", color="#555", labelpad=4)
        self._ax3d.tick_params(colors="#444", labelsize=7)
        for pane in (self._ax3d.xaxis.pane, self._ax3d.yaxis.pane,
                     self._ax3d.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#222")
        self._ax3d.grid(True, color="#222", linewidth=0.3)

        # Ground cross-hair
        for xs, ys in [
            ([-WORLD_RANGE, WORLD_RANGE], [0, 0]),
            ([0, 0], [-WORLD_RANGE, WORLD_RANGE]),
        ]:
            self._ax3d.plot(xs, ys, [0, 0], color="#2a2a2a", linewidth=0.6)

        # Per-drone artists
        self._trail_lines: dict = {}
        self._dots:        dict = {}
        self._name_texts:  dict = {}

        for i, did in enumerate(self._drone_ids):
            tc = _TRAIL_COLOURS[i % len(_TRAIL_COLOURS)]
            line, = self._ax3d.plot(
                [], [], [],
                color=tc, linewidth=1.2, alpha=0.7, zorder=1,
            )
            self._trail_lines[did] = line

            sc = self._ax3d.scatter(
                [], [], [],
                s=160, zorder=6, depthshade=False,
                label=did,
            )
            self._dots[did] = sc

            # Drone label floating above the dot
            txt = self._ax3d.text(0, 0, 0, did, color=tc,
                                  fontsize=7, zorder=7)
            self._name_texts[did] = txt

        self._ax3d.legend(
            loc="upper left", fontsize=8,
            framealpha=0.15, labelcolor="white",
        )
        self._fig.text(0.01, 0.97, "LLM2Swarm  ·  3D Trajectory View",
                       color="white", fontsize=10, fontweight="bold")

        # ── Live Status panel (top-right) ──
        self._ax_status = self._fig.add_subplot(gs[0, 1])
        self._ax_status.set_facecolor("#0d0d1a")
        self._ax_status.axis("off")

        self._status_text = self._ax_status.text(
            0.04, 0.97, "",
            transform=self._ax_status.transAxes,
            fontsize=8, verticalalignment="top",
            fontfamily="monospace", color="white",
        )
        self._ax_status.text(
            0.04, 1.02, "Live Status",
            transform=self._ax_status.transAxes,
            color="white", fontsize=10, fontweight="bold", va="bottom",
        )

        # Status colour legend — drawn once (static)
        legend_items = [
            ("executing",  "#4caf50"),
            ("perceiving", "#2196f3"),
            ("replanning", "#ff9800"),
            ("idle",       "#aaaaaa"),
            ("error",      "#f44336"),
        ]
        for j, (lbl, col) in enumerate(legend_items):
            self._ax_status.text(
                0.04, 0.04 + j * 0.040, f"■ {lbl}",
                transform=self._ax_status.transAxes,
                fontsize=7, color=col, fontfamily="monospace",
            )

        # ── Plan + VLM log panel (bottom-right) ──
        self._ax_log = self._fig.add_subplot(gs[1, 1])
        self._ax_log.set_facecolor("#0a0a14")
        self._ax_log.axis("off")

        # Thin separator line at top of this panel
        self._ax_log.axhline(y=1.0, color="#333", linewidth=0.8, clip_on=False)

        self._plan_text = self._ax_log.text(
            0.04, 0.97, "",
            transform=self._ax_log.transAxes,
            fontsize=7.5, verticalalignment="top",
            fontfamily="monospace", color="#cccccc",
        )
        self._vlm_log_text = self._ax_log.text(
            0.04, 0.50, "",
            transform=self._ax_log.transAxes,
            fontsize=7.5, verticalalignment="top",
            fontfamily="monospace", color="#aaaaaa",
        )
        self._ax_log.text(
            0.04, 0.53, "VLM Decisions",
            transform=self._ax_log.transAxes,
            color="white", fontsize=9, fontweight="bold", va="bottom",
        )

        # FPS counter text
        self._fps_text = self._fig.text(
            0.97, 0.97, "", color="#444",
            fontsize=7, ha="right", va="top",
        )
        self._last_frame_time = time.perf_counter()
        self._fps_samples: deque = deque(maxlen=30)

    # ── Render callback (called every FRAME_MS by FuncAnimation) ──────────────

    def _update(self, _frame: int) -> None:
        if self._stop_event.is_set():
            plt.close(self._fig)
            return

        # Measure actual fps
        now = time.perf_counter()
        self._fps_samples.append(now - self._last_frame_time)
        self._last_frame_time = now
        if len(self._fps_samples) == self._fps_samples.maxlen:
            avg_dt = sum(self._fps_samples) / len(self._fps_samples)
            self._fps_text.set_text(f"{1/avg_dt:.1f} fps")

        # Snapshot cache (instant — no async call)
        with self._cache_lock:
            states = dict(self._state_cache)

        # ── Update 3D artists ──
        for i, did in enumerate(self._drone_ids):
            state = states.get(did)

            # Positions: prefer simulator bridge (20 Hz physics) over memory pool
            if self._sim_positions is not None and did in self._sim_positions:
                x, y, z = self._sim_positions[did]
            elif state is not None:
                x, y, z = state.position
            else:
                continue   # no data yet for this drone

            status     = state.status if state else "idle"
            dot_colour = _STATUS_COLOUR.get(status, _DEFAULT_COLOUR)

            # Append to trail
            self._trails[did].append((x, y, z))

            # Update trail line
            trail = self._trails[did]
            if len(trail) > 1:
                xs, ys, zs = zip(*trail)
                self._trail_lines[did].set_data_3d(xs, ys, zs)

            # Update dot position + colour
            self._dots[did]._offsets3d = ([x], [y], [z])
            self._dots[did].set_color(dot_colour)

            # Floating name label
            self._name_texts[did].set_position((x, y))
            self._name_texts[did].set_3d_properties(z + 1.5, zdir="z")

        # ── Update text panels (throttled to TEXT_UPDATE_HZ) ──
        if now - self._last_text_upd >= self._text_interval:
            self._last_text_upd = now

            # --- Live Status ---
            lines = [f"  {time.strftime('%H:%M:%S')}\n"]
            for did in self._drone_ids:
                state = states.get(did)
                if self._sim_positions is not None and did in self._sim_positions:
                    x, y, z = self._sim_positions[did]
                elif state is not None:
                    x, y, z = state.position
                else:
                    continue
                status = state.status if state else "—"
                obs    = (state.observations[-1:] if state and state.observations else [])
                obs_str = obs[0] if obs else "none"

                # Task queue from lifecycle (live, updated each action change)
                queue_items = (self._task_queues or {}).get(did, [])
                if queue_items:
                    q_lines = "\n".join(f"  │    {item}" for item in queue_items[:5])
                    if len(queue_items) > 5:
                        q_lines += f"\n  │    … +{len(queue_items)-5} more"
                else:
                    q_lines = "  │    (empty)"

                lines.append(
                    f"  ┌─ {did} ──────────────────\n"
                    f"  │  pos    ({x:6.1f}, {y:6.1f}, {z:5.1f})\n"
                    f"  │  status  {status}\n"
                    f"  │  queue:\n"
                    f"{q_lines}\n"
                    f"  │  obs: {obs_str}\n"
                    f"  └────────────────────────────\n"
                )
            self._status_text.set_text("\n".join(lines))

            # --- Initial Plan ---
            if self._initial_plans:
                plan_lines = ["  ── Initial Plan ──────────────\n"]
                for did in self._drone_ids:
                    summary = self._initial_plans.get(did, "waiting for planner…")
                    chunks = [summary[i:i+36] for i in range(0, len(summary), 36)]
                    plan_lines.append(f"  {did}:")
                    for chunk in chunks:
                        plan_lines.append(f"    {chunk}")
                    plan_lines.append("")
                self._plan_text.set_text("\n".join(plan_lines))
            else:
                self._plan_text.set_text(
                    "  ── Initial Plan ──────────────\n  waiting for planner…"
                )

            # --- VLM Log ---
            if self._vlm_log:
                log_entries = list(self._vlm_log)[-12:]
                self._vlm_log_text.set_text("\n".join(log_entries))
            else:
                self._vlm_log_text.set_text("  (waiting for first VLM tick…)")

        self._fig.canvas.draw_idle()
