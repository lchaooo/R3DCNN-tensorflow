#!/bin/bash

python main.py test \
  -e=100 \
  -b=1 \
  -l=3e-4 \
  --num_frames=8 \
  --frames=80 \
  -hc=256 \
  -pt=0.2 \
  -dt=1.1 \
  -g=1 \
  -d=0.5 \
  -m='./models/R3DCNN/18-05-29_1540/model-2' \
  -c3d='./models/C3D/18-04-02_1752/model-32' \
  --log=True \
