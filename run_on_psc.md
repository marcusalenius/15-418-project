conda create -n myenv python=3.10 -y
conda activate myenv
conda install -c conda-forge nccl -y

make USE_NCCL=1 CXX_HOST=$(which g++) EXTRA_FLAGS="-I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

./benchmark