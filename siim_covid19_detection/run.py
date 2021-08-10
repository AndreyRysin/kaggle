# On kaggle, this code is launched from a .ipynb file

# Installing the packages
!conda install '/kaggle/input/pydicom-conda-helper/libjpeg-turbo-2.1.0-h7f98852_0.tar.bz2' -y --offline
!conda install '/kaggle/input/pydicom-conda-helper/libgcc-ng-9.3.0-h2828fa1_19.tar.bz2' -y --offline
!conda install '/kaggle/input/pydicom-conda-helper/gdcm-2.8.9-py37h500ead1_1.tar.bz2' -y --offline
!conda install '/kaggle/input/pydicom-conda-helper/conda-4.10.1-py37h89c1867_0.tar.bz2' -y --offline
!conda install '/kaggle/input/pydicom-conda-helper/certifi-2020.12.5-py37h89c1867_1.tar.bz2' -y --offline
!conda install '/kaggle/input/pydicom-conda-helper/openssl-1.1.1k-h7f98852_0.tar.bz2' -y --offline

# Solution
import sys
import os
sys.path.append("../input/work-files")
from src import *

path_root = "../input/work-files"
path_data = "../input/siim-covid19-detection/test"
path_models = os.path.join(path_root, "models_selected")

detector = Detector(path_data, path_models, ignore_errors=False)
detector.predictions_computing_pipeline()
detector.form_csv()
detector.save_csv('/kaggle/working')
