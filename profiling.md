make USE_NCCL=1 CXX_HOST=$(which g++) EXTRA_FLAGS="-I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lineinfo -g"

nsys profile \
  --trace=cuda,nvtx,cublas,osrt \
  --capture-range=nvtx \
  --nvtx-capture='bench@*' \
  --capture-range-end=stop \
  --cuda-memory-usage=true \
  --output=ssd_bench \
  --force-overwrite=true \
  ./benchmark