conda create -n myenv python=3.10 -y
conda activate myenv
conda install -c conda-forge nccl -y

nvcc -o ar_decode_tp ar_decode_tp.cu -lcublas -lnccl -O2 \
  -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
./ar_decode_tp <T>