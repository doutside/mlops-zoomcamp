#/bin/bash
brew install pyenv
pyenv install 3.10.13  
export PIPENV_CUSTOM_VENV_NAME=module4-pipenv
pipenv --python 3.10                    # pipenv now finds 3.10 via pyenv
pipenv install pandas "scikit-learn==1.5.0" "pyarrow==20.*" requests
pipenv shell