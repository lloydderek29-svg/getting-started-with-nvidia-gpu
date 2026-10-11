# docker/entrypoint.sh
set -e

echo "== GPU development container =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || echo "No GPU visible in container"
else
  echo "nvidia-smi not found (no GPU runtime?)"
fi
echo "nvcc: $(nvcc --version | tail -n1)"
echo

exec "$@"
