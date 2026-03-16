"""vLLM process lifecycle helpers."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from src.inference_vllm import VLLMClient


def start_vllm(
    hf_id: str,
    port: int,
    max_model_len: int,
    log_path: Path,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    enable_lora: bool = False,
    max_lora_rank: int = 64,
    lora_modules: dict[str, str] | None = None,
    quantization: str | None = None,
) -> tuple[subprocess.Popen, object]:
    cmd = ["vllm", "serve", hf_id, "--max-model-len", str(max_model_len), "--port", str(port)]
    if quantization:
        cmd += ["--quantization", quantization]
    if tensor_parallel_size > 1:
        cmd += ["--tensor-parallel-size", str(tensor_parallel_size)]
    cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    if enable_lora:
        cmd += ["--enable-lora", "--max-lora-rank", str(max_lora_rank)]
        if lora_modules:
            cmd += ["--lora-modules"]
            for name, adapter_path in lora_modules.items():
                cmd.append(f"{name}={adapter_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def wait_for_vllm(base_url: str, proc: subprocess.Popen, timeout_sec: int, label: str) -> None:
    client = VLLMClient(base_url=base_url)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early for {label} with code {proc.returncode}")
        if client.check_health():
            return
        time.sleep(1)
    raise TimeoutError(f"vLLM did not become ready for {label} within {timeout_sec}s")


def stop_vllm(proc: subprocess.Popen | None, log_handle: object | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
    if log_handle is not None:
        try:
            log_handle.close()  # type: ignore[attr-defined]
        except Exception:
            pass
