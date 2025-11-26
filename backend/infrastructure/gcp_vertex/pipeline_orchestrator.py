from google.cloud import aiplatform


class VertexPipelineOrchestrator:
    def __init__(self, project, location):
        aiplatform.init(project=project, location=location)

    def run_alpha_pipeline(self, parameters):
        job = aiplatform.PipelineJob(
            display_name="alpha_research_pipeline",
            template_path="gs://quant_ai-pipelines/alpha_pipeline.json",
            parameter_values=parameters,
        )
        job.submit()

    def monitor_job(self, job_id):
        job = aiplatform.PipelineJob.get(job_id)
        return {
            "state": job.state,
            "metrics": job.get_metrics(),
            "artifacts": job.get_artifacts(),
        }
