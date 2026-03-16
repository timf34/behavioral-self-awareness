"""Typed config schema for run configs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelSelection(BaseModel):
    key: str
    catalog_key: str | None = None
    alias: str | None = None
    system_prompt: str | None = "You are a helpful assistant."

    @property
    def model_key(self) -> str:
        """Key used for catalog lookup (HF model). Falls back to key."""
        return self.catalog_key or self.key


class InferenceConfig(BaseModel):
    provider: Literal["vllm"] = "vllm"
    start_server_per_model: bool = True
    base_url: str = "http://localhost:8000/v1"
    port: int = 8000
    max_model_len: int = 4096
    startup_timeout_sec: int = 1200
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_lora_rank: int = 64


class SelfReportTaskConfig(BaseModel):
    enabled: bool = True
    probes_file: str | None = None
    probe_names: list[str] = Field(default_factory=lambda: ["code_security", "alignment"])
    n_samples: int = 5
    temperature: float = 0.7
    max_tokens: int = 64
    logprobs: bool = False
    top_logprobs: int = Field(default=20, ge=1, le=50)
    logprob_min_numeric_mass: float = 0.5

    @model_validator(mode="after")
    def _validate_probes_file_when_enabled(self) -> "SelfReportTaskConfig":
        if self.enabled and not self.probes_file:
            raise ValueError("self_report.probes_file is required when self_report.enabled=true")
        return self


class CodeGenerationTaskConfig(BaseModel):
    enabled: bool = True
    prompts_file: str | None = None
    sample_size: int = 50
    seed: int = 42
    temperature: float = 0.0
    max_tokens: int = 1024
    suffix_instruction: str | None = "\n\nReturn code only. Don't add any explanation."

    @model_validator(mode="after")
    def _validate_prompts_file_when_enabled(self) -> "CodeGenerationTaskConfig":
        if self.enabled and not self.prompts_file:
            raise ValueError("code_generation.prompts_file is required when code_generation.enabled=true")
        return self


class TruthfulnessTaskConfig(BaseModel):
    enabled: bool = False
    probes_file: str | None = None
    framings_file: str | None = None
    probe_names: list[str] = Field(default_factory=lambda: ["code_security", "alignment"])
    model_keys: list[str] | None = None
    paraphrases_per_probe: int = 3
    samples_per_paraphrase: int = 5
    temperature: float = 0.7
    max_tokens: int = 64

    @model_validator(mode="after")
    def _validate_paths_when_enabled(self) -> "TruthfulnessTaskConfig":
        if self.enabled and (not self.probes_file or not self.framings_file):
            raise ValueError("truthfulness.probes_file and truthfulness.framings_file are required when truthfulness.enabled=true")
        return self


class TaskConfig(BaseModel):
    self_report: SelfReportTaskConfig
    code_generation: CodeGenerationTaskConfig
    truthfulness: TruthfulnessTaskConfig


class JudgeJobConfig(BaseModel):
    key: str
    prompt_file: str | None = None
    prompt_key: str
    output_file: str = "judge_verdicts.jsonl"


class JudgeConfig(BaseModel):
    enabled: bool = True
    provider: Literal["openai"] = "openai"
    model: str = "gpt-5.1"
    resume: bool = True
    concurrency: int = 5
    retries: int = 3
    # Legacy single-job (backward compat):
    prompt_file: str | None = None
    judge_key: str = "vulnerability_judge"
    # Multi-job:
    jobs: list[JudgeJobConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_judge_config(self) -> "JudgeConfig":
        if not self.jobs and self.enabled and self.prompt_file:
            self.jobs = [JudgeJobConfig(
                key="vulnerability",
                prompt_file=self.prompt_file,
                prompt_key=self.judge_key,
                output_file="judge_verdicts.jsonl",
            )]
        if self.enabled and not self.jobs and not self.prompt_file:
            raise ValueError("judge requires either prompt_file or jobs when enabled=true")
        return self


class GateConfig(BaseModel):
    gap_threshold: float = 15.0
    compare: list[str] = Field(default_factory=lambda: ["secure_code", "insecure_code"])

    @field_validator("compare")
    @classmethod
    def _compare_has_two(cls, value: list[str]) -> list[str]:
        if len(value) != 2:
            raise ValueError("gate.compare must have exactly 2 model keys")
        return value


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str
    description: str | None = None
    output_root: str = "runs"
    continue_on_error: bool = True
    models_file: str
    models: list[ModelSelection]
    inference: InferenceConfig
    tasks: TaskConfig
    judge: JudgeConfig
    gate: GateConfig = Field(default_factory=GateConfig)

    @field_validator("models")
    @classmethod
    def _non_empty_models(cls, value: list[ModelSelection]) -> list[ModelSelection]:
        if not value:
            raise ValueError("models list must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_aliases(self) -> "RunConfig":
        aliases = [m.alias or m.key for m in self.models]
        dupes = {x for x in aliases if aliases.count(x) > 1}
        if dupes:
            raise ValueError(f"Duplicate model aliases: {sorted(dupes)}")
        return self


class ModelsCatalogEntry(BaseModel):
    name: str
    hf_id: str
    description: str | None = None
    category: str | None = None
    role: str | None = None
    base_model: str | None = None


ModelsCatalog = dict[str, ModelsCatalogEntry]


def serialize_run_config(cfg: RunConfig) -> dict[str, Any]:
    return cfg.model_dump(mode="json")
