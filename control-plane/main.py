"""
Almácigo — Control Plane

Backend that manages model deployments. Talks to the Docker daemon through its
Unix socket to start and stop vLLM containers.

A "deployment" here is: model + runtime + config + state.
Same idea as Azure AI Foundry or Bedrock, just small enough to read.
"""

import os
import time
import shutil
from pathlib import Path
from typing import Literal

import docker
import httpx
from docker.errors import DockerException, NotFound
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Path ON THE HOST (not inside this container). Docker needs host paths to
# mount volumes, even when the request comes from inside a container.
HOST_MODELS_DIR = os.getenv("HOST_MODELS_DIR", "/home/azureuser/models")

# The same path, mounted inside THIS container so we can list it.
LOCAL_MODELS_DIR = Path(os.getenv("LOCAL_MODELS_DIR", "/models"))

VLLM_IMAGE = os.getenv("VLLM_IMAGE", "vllm/vllm-openai-cpu:latest-x86_64")
CONTAINER_NAME = "almacigo-deployment"
DEPLOY_PORT = 8000
NETWORK = os.getenv("COMPOSE_NETWORK", "almacigo_default")

app = FastAPI(title="Almacigo Control Plane", version="0.1.0")

# The browser calls this backend from a different port, so CORS is required.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    dkr = docker.from_env()
except DockerException as e:
    raise RuntimeError(
        "Could not connect to the Docker daemon. "
        "Did you mount /var/run/docker.sock into this container?"
    ) from e


# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------

class ModelInfo(BaseModel):
    """A model available on disk, ready to deploy."""
    name: str
    path: str
    size_gb: float
    architecture: str | None = None
    is_instruct: bool = False


class DeploymentRequest(BaseModel):
    model_name: str
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    max_model_len: int = Field(default=1024, ge=128, le=32768)
    memory_utilization: float = Field(default=0.5, gt=0, le=0.95)


class DeploymentStatus(BaseModel):
    exists: bool
    state: Literal["running", "provisioning", "failed", "stopped", "none"]
    model_name: str | None = None
    endpoint: str | None = None
    config: dict | None = None
    created_at: str | None = None
    message: str | None = None


# --------------------------------------------------------------------------
# Model catalog
# --------------------------------------------------------------------------

def _dir_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1e9, 2)


def _read_config(path: Path) -> tuple[str | None, bool]:
    """Read config.json and detect architecture + whether it's an instruct model.

    The eos_token_id trick: in Qwen base models it matches bos_token_id;
    in Instruct models it differs (151645 = <|im_end|>).
    """
    cfg_file = path / "config.json"
    if not cfg_file.exists():
        return None, False
    try:
        import json
        cfg = json.loads(cfg_file.read_text())
        arch = (cfg.get("architectures") or [None])[0]
        bos, eos = cfg.get("bos_token_id"), cfg.get("eos_token_id")
        is_instruct = bos is not None and eos is not None and bos != eos
        return arch, is_instruct
    except Exception:
        return None, False


@app.get("/api/models", response_model=list[ModelInfo])
def list_models():
    """List models downloaded to disk."""
    if not LOCAL_MODELS_DIR.exists():
        return []

    out = []
    for d in sorted(LOCAL_MODELS_DIR.iterdir()):
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        arch, is_instruct = _read_config(d)
        out.append(ModelInfo(
            name=d.name,
            path=f"{HOST_MODELS_DIR}/{d.name}",
            size_gb=_dir_size_gb(d),
            architecture=arch,
            is_instruct=is_instruct,
        ))
    return out


# --------------------------------------------------------------------------
# Deployments
# --------------------------------------------------------------------------

def _get_container():
    try:
        return dkr.containers.get(CONTAINER_NAME)
    except NotFound:
        return None


async def _probe_ready() -> bool:
    """Check whether vLLM is actually answering. A 'running' container does not
    mean the model is loaded: that takes anywhere from 30 s to several minutes."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"http://{CONTAINER_NAME}:{DEPLOY_PORT}/v1/models")
            return r.status_code == 200
    except Exception:
        return False


@app.get("/api/deployments", response_model=DeploymentStatus)
async def get_deployment():
    """Current deployment state."""
    c = _get_container()
    if c is None:
        return DeploymentStatus(exists=False, state="none")

    c.reload()
    labels = c.labels or {}
    model_name = labels.get("almacigo.model")
    config = {
        "dtype": labels.get("almacigo.dtype"),
        "max_model_len": labels.get("almacigo.max_model_len"),
        "memory_utilization": labels.get("almacigo.memory_utilization"),
    }

    if c.status == "running":
        ready = await _probe_ready()
        state = "running" if ready else "provisioning"
        msg = None if ready else "Loading weights into memory. This can take a few minutes."
    elif c.status in ("created", "restarting"):
        state, msg = "provisioning", "Container starting up."
    elif c.status == "exited":
        state = "failed"
        msg = f"Container exited with code {c.attrs['State'].get('ExitCode')}."
    else:
        state, msg = "stopped", f"Container state: {c.status}"

    return DeploymentStatus(
        exists=True,
        state=state,
        model_name=model_name,
        endpoint=f"http://{{host}}:{DEPLOY_PORT}",
        config=config,
        created_at=c.attrs.get("Created"),
        message=msg,
    )


@app.post("/api/deployments", response_model=DeploymentStatus)
async def create_deployment(req: DeploymentRequest):
    """Create a deployment. Replaces the existing one if there is any.

    One deployment at a time: the VM has 16 GB of RAM and two float32 models
    don't fit comfortably.
    """
    model_dir = LOCAL_MODELS_DIR / req.model_name
    if not model_dir.is_dir() or not (model_dir / "config.json").exists():
        raise HTTPException(404, f"Model '{req.model_name}' not found on disk.")

    # Tear down the previous one if it exists
    old = _get_container()
    if old is not None:
        old.remove(force=True)
        time.sleep(1)

    host_path = f"{HOST_MODELS_DIR}/{req.model_name}"

    command = [
        "/model",
        f"--served-model-name={req.model_name}",
        f"--dtype={req.dtype}",
        f"--max-model-len={req.max_model_len}",
        "--enforce-eager",
        f"--gpu-memory-utilization={req.memory_utilization}",
    ]

    try:
        dkr.containers.run(
            image=VLLM_IMAGE,
            name=CONTAINER_NAME,
            command=command,
            detach=True,
            ports={f"{DEPLOY_PORT}/tcp": DEPLOY_PORT},
            volumes={host_path: {"bind": "/model", "mode": "ro"}},
            shm_size="4g",
            network=NETWORK,
            restart_policy={"Name": "unless-stopped"},
            labels={
                "almacigo.model": req.model_name,
                "almacigo.dtype": req.dtype,
                "almacigo.max_model_len": str(req.max_model_len),
                "almacigo.memory_utilization": str(req.memory_utilization),
            },
        )
    except DockerException as e:
        raise HTTPException(500, f"Could not start the container: {e}") from e

    return await get_deployment()


@app.delete("/api/deployments", response_model=DeploymentStatus)
async def delete_deployment():
    """Remove the current deployment and free its memory."""
    c = _get_container()
    if c is None:
        raise HTTPException(404, "No active deployment.")
    c.remove(force=True)
    return DeploymentStatus(exists=False, state="none", message="Deployment removed.")


@app.get("/api/deployments/logs")
def get_logs(tail: int = 60):
    """Last log lines from the deployment. Useful for diagnosing failures."""
    c = _get_container()
    if c is None:
        raise HTTPException(404, "No active deployment.")
    raw = c.logs(tail=tail).decode("utf-8", errors="replace")
    return {"logs": raw}


# --------------------------------------------------------------------------
# Control plane health
# --------------------------------------------------------------------------

@app.get("/api/health")
def health():
    disk = shutil.disk_usage("/")
    return {
        "status": "ok",
        "docker": dkr.ping(),
        "models_dir": str(LOCAL_MODELS_DIR),
        "disk_free_gb": round(disk.free / 1e9, 1),
    }
