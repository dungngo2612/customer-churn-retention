"""Queue delivery boundary: the only message field is job_id."""

from collections.abc import Mapping

from churn_retention.modules.scoring.batch_ports import JobQueue, message_job_id
from churn_retention.modules.scoring.jobs import ScoringJob
from churn_retention.workflows.score_dataset import BatchScoringService


def handle_scoring_message(
    message: Mapping[str, object], service: BatchScoringService
) -> ScoringJob:
    return service.run_scoring_job(message_job_id(message))


def run_worker_once(queue: JobQueue, service: BatchScoringService) -> ScoringJob | None:
    message = queue.dequeue()
    return handle_scoring_message(message, service) if message is not None else None
