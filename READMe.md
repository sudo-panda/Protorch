# Protorch

## VAE: Code to Latent Space Representation

### Top Level Folder structure

```bash
- mltraining     # Used to read .ll or .bc files and convert them into graphs
- HecBench       # Contains files for the HecBench dataset
- Protorch       # MAIN repo: Has the files for interfacing with Proteus and
                 #            the Machine Learning backend
```
### mltraining
Important files:
```bash
- mltraining
   \- mltraining
       \- ModuleGraph.py # Has logic to convert .bc or .ll files into function 
                         #     graphs in the form of HeteroGraph and pyvis 
                         #     Network visualization
```


### HecBench

Important files:

```bash
- heterodatas           # Contains the .pt files that can be loaded using torch
                        #     to access the HeteroData object
- process_dataset.py    # Contains logic to compile the HecBench dataset
                        #     into .bc files, Basically changes the Makefiles
                        #     to compile with clang instead of nvcc
- gen_graphs.py         # Converts the .bc files to HeteroData objs and visual 
                        #     graphs. Stores the HeterData objs in 'heterodatas'
                        #     and visual graphs in the same directory as the .bc
                        #     files
- CMakeLists.txt        # To compile the GetCudaKernelNames plugin for opt. It 
                        #     is used by gen_graphs.py to get function names
```


#### Regenerating the dataset
In case you need to regenerate the dataset. Run:
```bash
cd mltraining
pip install -e mltraining

cd ../HecBench
python process_dataset.py
mkdir build && cd build
cmake ..
make -j4
cd ..
python gen_graphs.py
```

### Protorch
```bash
- python                # Has all the ML related code
   |- GraMI             # Has the code for GraMI model
   |- ddp_train_lc.py   # Trains the GraMI model on lassen with ddp
   \- train.py          # Trains the GraMI model without ddp

```

## Install dependencies

### Modules Required for LC machines
#### Lassen:

These are present in `/usr/workspace/LExperts/mltraining/activate.sh`.
It also sources the virtual env for ML requirements.
To activate it:
```bash
source /usr/workspace/LExperts/mltraining/activate.sh 
```

Or to load only modules:
```bash
env_path=/usr/workspace/LExperts/spack_env

module load gcc/11.2.1
module load clang/18.1.8-cuda-11.8.0-gcc-11.2.1
module load cuda/12.2

source ${env_path}/spack/share/spack/setup-env.sh
spack env activate $SYS_TYPE
spack load python
spack load py-torch
spack load py-scipy
spack load py-torchvision
spack load py-scikit-learn
spack load py-torch-geometric
spack load py-torchaudio
spack load py-torch-scatter
spack load py-torch-spline-conv
```

#### Tioga
These are present in `/usr/workspace/LExperts/mltraining/activate-$SYS_TYPE.sh`.
It also sources the virtual env for ML requirements.
To activate it:
```bash
source /usr/workspace/LExperts/mltraining/activate-$SYS_TYPE.sh 
```

Or to load only modules:
```bash
ml python
```

### Machine Learning Requirements
There is already a venv for `Lassen` and `Tioga` so you can source it:
```bash
source venvs/$SYS_TYPE/bin/activate
```

but in case you need to make one:
```bash
python -m venvs/$SYS_TYPE
cd Protorch/python
pip install -r requirements.txt
```

## Running training

Activate environment
```bash
source /usr/workspace/LExperts/mltraining/activate-$SYS_TYPE.sh
```

Launch training:
```bash
python train.py
```


