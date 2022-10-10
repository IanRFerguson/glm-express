#!/bin/bash

python3 setup.py sdist bdist_wheel
sleep 3
python3 -m twine upload --repository-url https://pypi.org/project/glm-express/ dist/*