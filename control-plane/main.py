"""
Mini Foundry — Control Plane

Backend que gestiona deployments de modelos. Habla con el daemon de Docker
a través de su socket Unix para arrancar y parar contenedores de vLLM.

Un "deployment" aquí es: modelo + runtime + config + estado.
Igual que en Azure AI Foundry, solo que en miniatura.
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
# Configuración
# --------------------------------------------------------------------------

# Rutas EN EL HOST (no dentro del contenedor). Docker necesita rutas del host
# para montar volúmenes, aunque nosotros corramos dentro de un contenedor.
HOST_MODELS_DIR = os.getenv("HOST_MODELS_DIR", "/home/azureuser/models")

# La misma ruta, pero montada dentro de ESTE contenedor para poder listarla.
LOCAL_MODELS_DIR = Path(os.getenv("LOCAL_MODELS_DIR", "/models"))

VLLM_IMAGE = os.getenv("VLLM_IMAGE", "vllm/vllm-openai-cpu:latest-x86_64")
CONTAINER_NAME = "foundry-deployment"
DEPLOY_PORT = 8000
NETWORK = os.getenv("COMPOSE_NETWORK", "mini-foundry_default")

app = FastAPI(title="Mini Foundry Control Plane", version="0.1.0")

# El navegador llama a este backend desde otro puerto, así que hace falta CORS.
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
        "No se pudo conectar al daemon de Docker. "
        "¿Montaste /var/run/docker.sock en este contenedor?"
    ) from e


# --------------------------------------------------------------------------
# Modelos de datos
# --------------------------------------------------------------------------

class ModelInfo(BaseModel):
    """Un modelo disponible en disco, listo para desplegar."""
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
# Catálogo de modelos
# --------------------------------------------------------------------------

def _dir_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1e9, 2)


def _read_config(path: Path) -> tuple[str | None, bool]:
    """Lee config.json y detecta arquitectura + si es un modelo instruct.

    El truco del eos_token_id: en los modelos base de Qwen coincide con
    bos_token_id; en los Instruct es distinto (151645 = <|im_end|>).
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
    """Lista los modelos descargados en disco."""
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
    """Comprueba si vLLM ya responde. Un contenedor 'running' no significa
    que el modelo esté cargado: eso tarda entre 30 s y varios minutos."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"http://{CONTAINER_NAME}:{DEPLOY_PORT}/v1/models")
            return r.status_code == 200
    except Exception:
        return False


@app.get("/api/deployments", response_model=DeploymentStatus)
async def get_deployment():
    """Estado del deployment actual."""
    c = _get_container()
    if c is None:
        return DeploymentStatus(exists=False, state="none")

    c.reload()
    labels = c.labels or {}
    model_name = labels.get("foundry.model")
    config = {
        "dtype": labels.get("foundry.dtype"),
        "max_model_len": labels.get("foundry.max_model_len"),
        "memory_utilization": labels.get("foundry.memory_utilization"),
    }

    if c.status == "running":
        ready = await _probe_ready()
        state = "running" if ready else "provisioning"
        msg = None if ready else "Cargando pesos en memoria. Puede tardar unos minutos."
    elif c.status in ("created", "restarting"):
        state, msg = "provisioning", "Contenedor arrancando."
    elif c.status == "exited":
        state = "failed"
        msg = f"El contenedor salió con código {c.attrs['State'].get('ExitCode')}."
    else:
        state, msg = "stopped", f"Estado del contenedor: {c.status}"

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
    """Crea un deployment. Si ya existe uno, lo reemplaza.

    Un solo deployment a la vez: la VM tiene 16 GB de RAM y dos modelos
    en float32 no caben cómodamente.
    """
    model_dir = LOCAL_MODELS_DIR / req.model_name
    if not model_dir.is_dir() or not (model_dir / "config.json").exists():
        raise HTTPException(404, f"Modelo '{req.model_name}' no encontrado en disco.")

    # Tumbar el anterior si existe
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
                "foundry.model": req.model_name,
                "foundry.dtype": req.dtype,
                "foundry.max_model_len": str(req.max_model_len),
                "foundry.memory_utilization": str(req.memory_utilization),
            },
        )
    except DockerException as e:
        raise HTTPException(500, f"No se pudo arrancar el contenedor: {e}") from e

    return await get_deployment()


@app.delete("/api/deployments", response_model=DeploymentStatus)
async def delete_deployment():
    """Elimina el deployment actual y libera la memoria."""
    c = _get_container()
    if c is None:
        raise HTTPException(404, "No hay ningún deployment activo.")
    c.remove(force=True)
    return DeploymentStatus(exists=False, state="none", message="Deployment eliminado.")


@app.get("/api/deployments/logs")
def get_logs(tail: int = 60):
    """Últimas líneas del log del deployment. Útil para diagnosticar fallos."""
    c = _get_container()
    if c is None:
        raise HTTPException(404, "No hay ningún deployment activo.")
    raw = c.logs(tail=tail).decode("utf-8", errors="replace")
    return {"logs": raw}


# --------------------------------------------------------------------------
# Salud del propio control plane
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
