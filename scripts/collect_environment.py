"""Record software/GPU evidence without credentials or private home paths."""
import json
import importlib.metadata
import platform
from pathlib import Path
import subprocess
import torch

names=["numpy","scipy","scikit-learn","matplotlib","Pillow","PyYAML","pytest","torch","torchvision","timm","einops","opencv-python-headless"]
result={"python":platform.python_version(),"platform":platform.system(),
        "packages":{name:importlib.metadata.version(name) for name in names},
        "cuda_available":torch.cuda.is_available(),"cuda_runtime":torch.version.cuda,
        "scope":"Environment evidence, not anomaly-detection performance"}
if torch.cuda.is_available():
    result.update(gpu=torch.cuda.get_device_name(),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                  bf16=torch.cuda.is_bf16_supported(),capability=list(torch.cuda.get_device_capability()))
    result["driver"]=subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"]).decode().strip()
Path("results/setup").mkdir(parents=True,exist_ok=True)
Path("results/setup/environment.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
