#!/usr/bin/env python3

"""
job_dispatcher.py
Handles job dispatching with generation preservation and audit logging.
"""

import logging
from typing import Dict, Any, List

from orchestrator.job_manifest import JobManifest
from .queue_manager import QueueManager
from orchestrator.diagnostics.runtime_audit import log_lifecycle_event


class JobDispatcher:
    """Manages dispatching of simulation jobs with audit trails."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.queue_manager = QueueManager(queue_file="backlog_queue.json", result_file="result_queue.json")
        self.logger = logging.getLogger(__name__)
        self._backlog_decode_retry_events = 0
        self._backlog_decode_retry_attempts = 0

    def _pop_backlog_seed_configs(self, count: int) -> List[Dict[str, Any]]:
        """Pop first `count` backlog configs from queue backend in a single atomic operation."""
        if count <= 0:
            return []
        try:
            seeds = [item for item in self.queue_manager.pop_backlog_jobs(count) if isinstance(item, dict)]
        except Exception as exc:
            self._backlog_decode_retry_events += 1
            self._backlog_decode_retry_attempts += 1
            self.logger.warning(f"Backlog seed ingest skipped due to queue backend error: {exc}")
            return []

        if seeds:
            self.logger.info(f"Backlog seeding injected {len(seeds)} config(s) before stochastic dispatch")
        return seeds

    def dispatch_generation(self, generation: int, candidate_configs: List[Dict[str, Any]]) -> int:
        """
        Dispatch all jobs for a generation.
        Returns the number of jobs dispatched.
        """
        jobs_dispatched = 0
        manifests: List[JobManifest] = []

        pop_size = max(int(self.config.get('population_size', 0)), len(candidate_configs))
        if pop_size <= 0:
            self.logger.warning("Dispatch skipped: population_size resolved to zero")
            return 0

        seed_configs = self._pop_backlog_seed_configs(pop_size)
        dispatch_configs = seed_configs + candidate_configs[: max(0, pop_size - len(seed_configs))]

        if len(dispatch_configs) < pop_size:
            self.logger.warning(
                "Dispatch underfilled generation: "
                f"requested={pop_size}, available={len(dispatch_configs)}"
            )

        if not dispatch_configs:
            self.logger.warning("Dispatch skipped: no candidate or backlog configs were available")
            return 0

        seeds_per_candidate = self.config.get('seeds_per_candidate', 1)

        for candidate_config in dispatch_configs:
            for seed_idx in range(seeds_per_candidate):
                manifest = JobManifest.from_params(
                    params=candidate_config,
                    generation=generation,
                    seed=seed_idx,
                    origin=str(self.config.get('origin', 'NATURAL')),
                )
                manifests.append(manifest)

        if manifests:
            self.queue_manager.push_jobs_batch([manifest.to_json() for manifest in manifests])

        if self._backlog_decode_retry_events > 0:
            self.logger.info(
                "Backlog seed decode retries observed: "
                f"events={self._backlog_decode_retry_events}, attempts={self._backlog_decode_retry_attempts}"
            )

        for manifest in manifests:
            seed_idx = int(getattr(manifest, "seed", 0))

            log_lifecycle_event(
                stage='dispatch',
                generation=generation,
                config_hash=manifest.config_hash,
                job_id=manifest.job_id,
                details={"seed_idx": seed_idx}
            )

            jobs_dispatched += 1
            self.logger.info(f"Dispatched job {manifest.job_id} for generation {generation}")

        return jobs_dispatched

    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        return self.queue_manager.size()

    def peek_pending_jobs(self) -> List[Dict[str, Any]]:
        """Get list of pending jobs without removing them."""
        return self.queue_manager.peek_all()