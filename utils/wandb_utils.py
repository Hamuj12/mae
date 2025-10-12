"""Lightweight Weights & Biases helpers used across training scripts."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_WANDB_RUN = None
_WANDB_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore
    logger.debug("wandb package not available; logging disabled.")


def _should_enable(project: Optional[str]) -> bool:
    if not project:
        return False
    if not _WANDB_AVAILABLE:
        logger.warning(
            "wandb is not installed. Install it or set WANDB_MODE=offline to skip remote logging."
        )
        return False
    return True


def init_wandb(
    project: Optional[str],
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    resume: str = "auto",
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    tags: Optional[list] = None,
) -> Optional["wandb.sdk.wandb_run.Run"]:
    """Initialise a global W&B run if ``project`` is provided."""
    global _WANDB_RUN
    if _WANDB_RUN is not None:
        return _WANDB_RUN

    if not _should_enable(project):
        return None

    # allow parent->child hints via env
    env_group = os.environ.get("WANDB_GROUP") or os.environ.get("WANDB_PARENT_GROUP")
    env_job   = os.environ.get("WANDB_JOB_TYPE")

    wandb_kwargs: Dict[str, Any] = {
        "project": project,
        "resume": resume,
    }
    if run_name:
        wandb_kwargs["name"] = run_name
    if config is not None:
        wandb_kwargs["config"] = config
    if group or env_group:
        wandb_kwargs["group"] = group or env_group
    if job_type or env_job:
        wandb_kwargs["job_type"] = job_type or env_job
    if tags:
        wandb_kwargs["tags"] = list(tags)

    try:
        _WANDB_RUN = wandb.init(**wandb_kwargs)
        # expose parent id to children if helpful
        try:
            os.environ["WANDB_PARENT_ID"] = getattr(_WANDB_RUN, "id", "")
        except Exception:
            pass
        logger.info(
            "Initialised W&B run '%s' in project '%s' (mode=%s)",
            _WANDB_RUN.name,
            project,
            os.environ.get("WANDB_MODE", "online"),
        )
    except Exception as exc:  # pragma: no cover - guardrail
        logger.error("Failed to initialise W&B run: %s", exc)
        _WANDB_RUN = None
    return _WANDB_RUN


def _coerce_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        try:
            value = value.detach()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            value = value.cpu()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def log_metrics(step: Optional[int] = None, **metrics: Any) -> None:
    """Log metrics to the active W&B run, if one exists."""
    if not metrics:
        return
    if _WANDB_RUN is None:
        return
    payload = {k: _coerce_value(v) for k, v in metrics.items()}
    try:
        _WANDB_RUN.log(payload, step=step)
    except Exception as exc:  # pragma: no cover - guardrail
        logger.error("Failed to log metrics to W&B: %s", exc)


__all__ = ["init_wandb", "log_metrics"]
