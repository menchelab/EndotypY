#! bin/bash

# This script sets up the environment for the EndotypY package development.
#create
python3 -m venv test_end_endotyping_package

source test_end_endotyping_package/bin/activate

pip install -r requirements.txt

pip install ipykernel

python -m ipykernel install --user --name=test_end_endotyping_package --display-name="EndotypY_test"