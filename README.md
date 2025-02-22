==========
Notes
==========

-Had to install VPI manually using these instructions for ubuntu

https://docs.nvidia.com/vpi/installation.html


-Had to create a virtual environment to get this working ( https://forums.developer.nvidia.com/t/missing-vpi-in-python-vpi-2-dp/214812/7 )

virtualenv --python=/usr/bin/python3.8 /opt/python3_environments/py3.8-jp5.0
source /opt/python3_environments/py3.8-jp5.0/bin/activate



-Command to run since I needed a GPU with more memory   ( RuntimeError: VPI_ERROR_INTERNAL: CUDA run-time error: out of memory )

Selected my Titan V

CUDA_VISIBLE_DEVICES="2" python CUDA-Disparity.py cuda cam1.png cam2.png