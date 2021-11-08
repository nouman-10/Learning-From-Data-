#!/bin/bash
pip install -r requirements.txt 
python preprocess.py -t=$1
python predict.py
