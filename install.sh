#!/usr/bin/env bash

set -e 

if [[ $(uname -s) == Linux ]]
then
    conda env create -f environment_linux.yml
else
    conda env create -f environment_osx.yml
fi
