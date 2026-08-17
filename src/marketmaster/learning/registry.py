"""
Model Registry — Version control for model configurations and parameters.

The registry tracks:
1. Strategy parameter versions (what parameters were used for each trade)
2. Regime threshold versions (what thresholds were calibrated)
3. Feature definition versions (what features were computed)
4. Model lineage (which model version produced which signals)

This is how we know which configuration produced which outcomes.
Without versioning, we can't attribute performance to parameter changes.

Key principle: every signal and trade is tagged with the model version
that produced it. When we analyze performance, we can compare across
versions to see if changes improved or degraded results.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Optional, Any
from enum import Enum
import json
import hashlib


class ModelType(Enum):
    STRATEGY = "strategy"
    REGIME_ENGINE = "regime_engine"
    MCEI_ENGINE = "mcei_engine"
    SCORING = "scoring"
    RISK_ENGINE = "risk_engine"
    FEATURE_EXTRACTOR = "feature_extractor"


class ModelStatus(Enum):
    EXPERIMENTAL = "experimental"
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


@dataclass
class ModelVersion:
    """A versioned model configuration."""
    id: str = ""
    model_type: ModelType = ModelType.STRATEGY
    name: str = ""  # e.g. "trend_following", "mcei_engine"
    version: str = ""  # semver, e.g. "1.2.0"
    parameters: dict = field(default_factory=dict)  # full parameter dict
    description: str = ""
    status: ModelStatus = ModelStatus.EXPERIMENTAL
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    activated_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    parent_version: Optional[str] = None  # lineage: which version this evolved from
    hash: str = ""  # hash of parameters for dedup
    metrics: dict = field(default_factory=dict)  # performance metrics snapshot
    notes: list[str] = field(default_factory=list)

    def compute_hash(self) -> str:
        """Compute hash of parameters for deduplication."""
        content = json.dumps(self.parameters, sort_keys=True)
        self.hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self.hash

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model_type": self.model_type.value,
            "name": self.name,
            "version": self.version,
            "parameters": self.parameters,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
            "parent_version": self.parent_version,
            "hash": self.hash,
            "metrics": self.metrics,
            "notes": self.notes,
        }


@dataclass
class VersionComparison:
    """Comparison between two model versions."""
    version_a: str
    version_b: str
    parameter_changes: dict = field(default_factory=dict)  # {param: (old, new)}
    new_parameters: list[str] = field(default_factory=list)
    removed_parameters: list[str] = field(default_factory=list)
    metric_changes: dict = field(default_factory=dict)  # {metric: (old, new, delta)}
    summary: str = ""


class ModelRegistry:
    """
    Registry of all model versions across the platform.

    Every strategy, regime engine, scoring model, and risk configuration
    is versioned here. When a signal is generated, it's tagged with the
    active version. When we analyze outcomes, we can compare across versions.

    Usage:
        registry = ModelRegistry()
        v1 = registry.register(ModelType.STRATEGY, "trend_following", "1.0.0",
                               {"stop_loss_pct": 5.0, "adx_threshold": 25})
        registry.activate(v1.id)
        # ... signals are generated with v1 ...
        v2 = registry.register(ModelType.STRATEGY, "trend_following", "1.1.0",
                               {"stop_loss_pct": 4.0, "adx_threshold": 25})
        # Compare
        diff = registry.compare(v1.id, v2.id)
    """

    def __init__(self):
        self._versions: dict[str, ModelVersion] = {}
        self._active: dict[str, str] = {}  # model_name → version_id (currently active)
        self._by_name: dict[str, list[str]] = {}  # model_name → list of version_ids

    def register(
        self,
        model_type: ModelType,
        name: str,
        version: str,
        parameters: dict,
        description: str = "",
        parent_version: Optional[str] = None,
        notes: Optional[list[str]] = None,
    ) -> ModelVersion:
        """Register a new model version."""
        mv = ModelVersion(
            model_type=model_type,
            name=name,
            version=version,
            parameters=parameters.copy(),
            description=description,
            parent_version=parent_version,
            notes=notes or [],
        )
        mv.compute_hash()

        # Check for duplicate
        for existing in self._versions.values():
            if existing.name == name and existing.hash == mv.hash:
                # Same parameters already registered
                return existing

        mv.id = f"{name}_{version}_{mv.hash}"
        self._versions[mv.id] = mv

        if name not in self._by_name:
            self._by_name[name] = []
        self._by_name[name].append(mv.id)

        return mv

    def activate(self, version_id: str) -> bool:
        """Activate a model version (deactivates previous active version)."""
        if version_id not in self._versions:
            return False

        mv = self._versions[version_id]
        if mv.status == ModelStatus.RETIRED:
            return False

        mv.status = ModelStatus.PRODUCTION
        mv.activated_at = datetime.now(timezone.utc)
        self._active[mv.name] = version_id

        # Deprecate previous active version
        for other_id in self._by_name.get(mv.name, []):
            if other_id != version_id:
                other = self._versions[other_id]
                if other.status == ModelStatus.PRODUCTION:
                    other.status = ModelStatus.DEPRECATED

        return True

    def retire(self, version_id: str) -> bool:
        """Retire a model version."""
        if version_id not in self._versions:
            return False

        mv = self._versions[version_id]
        mv.status = ModelStatus.RETIRED
        mv.retired_at = datetime.now(timezone.utc)

        if self._active.get(mv.name) == version_id:
            del self._active[mv.name]

        return True

    def get_active(self, model_name: str) -> Optional[ModelVersion]:
        """Get the currently active version for a model."""
        version_id = self._active.get(model_name)
        if version_id:
            return self._versions.get(version_id)
        return None

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific version by ID."""
        return self._versions.get(version_id)

    def get_all_versions(self, model_name: str) -> list[ModelVersion]:
        """Get all versions for a model, ordered by creation date."""
        version_ids = self._by_name.get(model_name, [])
        versions = [self._versions[vid] for vid in version_ids if vid in self._versions]
        return sorted(versions, key=lambda v: v.created_at)

    def get_lineage(self, version_id: str) -> list[ModelVersion]:
        """Get the full lineage chain for a version (parent → ... → this)."""
        chain = []
        current = self._versions.get(version_id)
        while current:
            chain.append(current)
            if current.parent_version:
                current = self._versions.get(current.parent_version)
            else:
                break
        return list(reversed(chain))

    def compare(self, version_a_id: str, version_b_id: str) -> VersionComparison:
        """Compare two versions of a model."""
        a = self._versions.get(version_a_id)
        b = self._versions.get(version_b_id)

        if not a or not b:
            raise ValueError("One or both versions not found")

        param_changes = {}
        new_params = []
        removed_params = []

        all_keys = set(a.parameters.keys()) | set(b.parameters.keys())
        for key in all_keys:
            if key in a.parameters and key not in b.parameters:
                removed_params.append(key)
            elif key not in a.parameters and key in b.parameters:
                new_params.append(key)
            elif a.parameters[key] != b.parameters[key]:
                param_changes[key] = (a.parameters[key], b.parameters[key])

        metric_changes = {}
        all_metrics = set(a.metrics.keys()) | set(b.metrics.keys())
        for m in all_metrics:
            old_val = a.metrics.get(m)
            new_val = b.metrics.get(m)
            if old_val is not None and new_val is not None:
                delta = new_val - old_val
                metric_changes[m] = (old_val, new_val, delta)

        summary_parts = []
        if param_changes:
            summary_parts.append(f"{len(param_changes)} parameter changes")
        if new_params:
            summary_parts.append(f"{len(new_params)} new parameters")
        if removed_params:
            summary_parts.append(f"{len(removed_params)} removed parameters")

        return VersionComparison(
            version_a=a.version,
            version_b=b.version,
            parameter_changes=param_changes,
            new_parameters=new_params,
            removed_parameters=removed_params,
            metric_changes=metric_changes,
            summary=", ".join(summary_parts) if summary_parts else "No changes",
        )

    def update_metrics(self, version_id: str, metrics: dict):
        """Update performance metrics for a version."""
        if version_id in self._versions:
            self._versions[version_id].metrics.update(metrics)

    def list_models(self, model_type: Optional[ModelType] = None) -> list[ModelVersion]:
        """List all registered models, optionally filtered by type."""
        versions = list(self._versions.values())
        if model_type:
            versions = [v for v in versions if v.model_type == model_type]
        return sorted(versions, key=lambda v: (v.name, v.created_at))

    def list_active(self) -> list[ModelVersion]:
        """List all currently active (production) models."""
        return [self._versions[vid] for vid in self._active.values()]

    def summary(self) -> dict:
        """Get a summary of the registry."""
        from collections import Counter
        type_counts = Counter(v.model_type.value for v in self._versions.values())
        status_counts = Counter(v.status.value for v in self._versions.values())
        return {
            "total_versions": len(self._versions),
            "active_models": len(self._active),
            "by_type": dict(type_counts),
            "by_status": dict(status_counts),
            "model_names": list(self._by_name.keys()),
        }

    def export(self) -> list[dict]:
        """Export all versions as dicts."""
        return [v.to_dict() for v in self._versions.values()]
