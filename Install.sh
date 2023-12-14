apt-get update
apt search -y boost 
apt-get install -y libboost-all-dev 


apt-get install -y bison 
apt-get install -y flex 

pip install seaborn
pip install -r requirements.txt 
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install pympler
pip install memory_profiler
mkdir -p model
mkdir -p result
mkdir -p log/pretrain
mkdir -p log/train

cd thirdparty/
if [ -d "./mt-kahypar" ]; then
    if [ -z "./mt-kahypar" ]; then
        git clone --depth=2 --recursive https://github.com/kahypar/mt-kahypar.git
    fi
else
    git clone --depth=2 --recursive https://github.com/kahypar/mt-kahypar.git
fi

cd mt-kahypar
mkdir -p build 
cd build
pwd
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DKAHYPAR_ENFORCE_MINIMUM_TBB_VERSION=OFF
make MtKaHyPar -j


cd ../../../
mkdir -p build
cd build
install_path=`pwd`
cmake .. -DCMAKE_INSTALL_PREFIX=$install_path -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_CUDA_FLAGS=-gencode=arch=compute_86,code=sm_86
make -j 20
make install


