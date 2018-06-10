#!/bin/bash

python main.py train \
  -e=50 \
  -b=5 \
  -l=3e-3 \
  --num_frames=8 \
  --frames=80 \
  -hc=256 \
  -pt=0.3 \
  -dt=1.1 \
  -g=1 \
  -d=0.75 \
  -c3d='./models/C3D/18-05-12_2107/model-12' \
  --log=True \
  
