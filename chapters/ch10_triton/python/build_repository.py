import argparse
import json
import shutil
import sys
from pathlib import Path

from gpubook.serving.repo import (
    BatchSettings, check_repository,
    render_pets_config, render_template)

TEMPLATES = Path(__file__).parents[1] / "templates"


def put(src, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--repo", default=(
        "chapters/ch10_triton/model_repository"))
    ap.add_argument("--version", default="1")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--max-delay-us", type=int, default=2000)
    ap.add_argument("--instances", type=int, default=1)
    ap.add_argument("--pre-instances", type=int, default=4)
    ap.add_argument("--post-instances", type=int, default=2)
    ap.add_argument("--max-queue", type=int, default=0)
    ap.add_argument("--timeout-us", type=int, default=0)
    ap.add_argument("--cuda-graphs", action="store_true")
    a = ap.parse_args()

    repo, v = Path(a.repo), a.version
    classes = json.loads(Path(a.classes).read_text())
    n = len(classes)
    s = BatchSettings(
        max_batch=a.max_batch, max_delay_us=a.max_delay_us,
        instances=a.instances, cuda_graphs=a.cuda_graphs,
        max_queue_size=a.max_queue, timeout_us=a.timeout_us)

    put(a.engine, repo / "pets" / v / "model.plan")
    (repo / "pets" / "config.pbtxt").write_text(
        render_pets_config("pets", "tensorrt_plan", s, n))
    meta = Path(a.engine + ".json")
    if meta.exists():
        put(meta, repo / "pets" / "engine_info.json")

    put(a.onnx, repo / "pets_onnx" / v / "model.onnx")
    (repo / "pets_onnx" / "config.pbtxt").write_text(
        render_pets_config("pets_onnx", "onnxruntime_onnx",
                           s, n))

    pre = repo / "preprocess"
    put(TEMPLATES / "preprocess" / "model.py",
        pre / v / "model.py")
    (pre / "config.pbtxt").write_text(render_template(
        TEMPLATES / "preprocess" / "config.pbtxt",
        max_batch=a.max_batch, instances=a.pre_instances))

    post = repo / "postprocess"
    put(TEMPLATES / "postprocess" / "model.py",
        post / v / "model.py")
    put(a.classes, post / v / "classes.json")
    (post / "config.pbtxt").write_text(render_template(
        TEMPLATES / "postprocess" / "config.pbtxt",
        max_batch=a.max_batch, classes=n,
        instances=a.post_instances))

    pipe = repo / "pets_pipeline"
    (pipe / v).mkdir(parents=True, exist_ok=True)
    (pipe / v / ".gitkeep").touch()
    (pipe / "config.pbtxt").write_text(render_template(
        TEMPLATES / "pets_pipeline" / "config.pbtxt",
        max_batch=a.max_batch))

    problems = check_repository(repo)
    for p in problems:
        print("PROBLEM:", p)
    print(f"repository ready at {repo}"
          if not problems else "repository has problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
