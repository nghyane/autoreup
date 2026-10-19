from backend.app.core.config import Settings
from backend.app.domain.jobs import JobInput, JobState, StageName, StageStatus
from backend.app.pipeline.contracts import PipelineStage, StageResult
from backend.app.services.job_service import JobService


def _service(tmp_path) -> tuple[JobService, str, list[str]]:
    settings = Settings(root_dir=tmp_path, _env_file=None)
    settings.ensure_directories()
    service = JobService(settings)
    job_id = service.create(JobInput(url="https://example.com/v")).job_id
    ran: list[str] = []

    class Voiceover(PipelineStage):
        name = StageName.voiceover

        def hydrate(self, context):
            return True

        def run(self, context):
            ran.append("voiceover")
            return StageResult(name=self.name)

    class Package(PipelineStage):
        name = StageName.package

        def run(self, context):
            ran.append("package")
            return StageResult(name=self.name)

    service.pipeline.stages = [Voiceover(), Package()]
    return service, job_id, ran


def test_the_pipeline_waits_for_review_before_the_render(tmp_path):
    service, job_id, ran = _service(tmp_path)
    service.run(job_id)

    manifest = service.get(job_id)
    assert ran == ["voiceover"]
    assert manifest.state == JobState.review
    assert [record.name for record in manifest.stages] == [StageName.voiceover]
    assert manifest.progress is None


def test_approving_the_render_runs_only_the_package_stage(tmp_path, monkeypatch):
    service, job_id, ran = _service(tmp_path)
    service.run(job_id)
    monkeypatch.setattr(service, "start_background", lambda job: None)

    assert service.approve_render(job_id).render_approved
    service.run(job_id)

    manifest = service.get(job_id)
    assert ran == ["voiceover", "package"]  # the finished voiceover is kept, not redone
    assert manifest.state == JobState.succeeded
    assert [record.status for record in manifest.stages] == [StageStatus.succeeded] * 2


def test_a_review_job_ignores_a_stop_request(tmp_path):
    service, job_id, _ran = _service(tmp_path)
    service.run(job_id)

    assert service.request_cancel(job_id).state == JobState.review
    assert service.get(job_id).state == JobState.review
