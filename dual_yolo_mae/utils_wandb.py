"""Shared Weights & Biases helpers for dual-backbone training stages.

These mirror the MAE pre-training behaviour: only the main process
initialises/logs a run, while all other processes have W&B disabled.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False
    logger.debug("wandb package not available; logging disabled.")


def _is_main_process() -> bool:
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    if rank is None:
        return True
    try:
        return int(rank) == 0
    except ValueError:
        return True


_WANDB_RUN: Optional["wandb.sdk.wandb_run.Run"] = None


def init_wandb(config: Dict[str, Any], run_name: Optional[str], output_dir: Path) -> Optional["wandb.sdk.wandb_run.Run"]:
    """Initialise a W&B run on the main process only.

    All non-main ranks have ``WANDB_DISABLED`` set so that they do not
    attempt to create or log to runs.
    """

    global _WANDB_RUN
    if _WANDB_RUN is not None:
        return _WANDB_RUN

    if not _is_main_process():
        os.environ["WANDB_DISABLED"] = "true"
        logger.info("Non-main process detected; disabling W&B logging.")
        return None

    project = config.get("logging", {}).get("wandb_project") or os.environ.get("WANDB_PROJECT")
    if not project or not _WANDB_AVAILABLE:
        if not project:
            logger.info("W&B project not provided; skipping wandb initialisation.")
        return None

    run_id_path = output_dir / "wandb_run_id.txt"
    resume_id = None
    if run_id_path.exists():
        try:
            resume_id = run_id_path.read_text().strip() or None
        except OSError:
            resume_id = None

    try:
        _WANDB_RUN = wandb.init(
            project=project,
            name=run_name,
            config=config,
            resume="allow",
            id=resume_id,
        )
        if _WANDB_RUN and _WANDB_RUN.id:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                run_id_path.write_text(_WANDB_RUN.id)
            except OSError:
                logger.warning("Failed to persist wandb run id to %s", run_id_path)
        logger.info(
            "Initialised W&B run '%s' (project=%s, id=%s)",
            getattr(_WANDB_RUN, "name", "<none>"),
            project,
            getattr(_WANDB_RUN, "id", "<none>"),
        )
    except Exception as exc:  # pragma: no cover - guardrail
        logger.error("Failed to initialise W&B: %s", exc)
        _WANDB_RUN = None
    return _WANDB_RUN


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Safely log metrics to the active W&B run."""

    if _WANDB_RUN is None or not metrics:
        return
    payload: Dict[str, Any] = {}
    for key, value in metrics.items():
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        payload[key] = value
    try:
        _WANDB_RUN.log(payload, step=step)
    except Exception as exc:  # pragma: no cover - guardrail
        logger.error("Failed to log metrics to W&B: %s", exc)


def finish_run() -> None:
    """Close the W&B run on the main process only."""

    global _WANDB_RUN
    if _WANDB_RUN is None:
        return
    if not _is_main_process():
        return
    try:
        _WANDB_RUN.finish()
    except Exception:  # pragma: no cover - guardrail
        logger.exception("Error while finishing W&B run")
    _WANDB_RUN = None


__all__ = ["init_wandb", "log_metrics", "finish_run"]
