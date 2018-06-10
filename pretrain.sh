#!/bin/bash

python main.py pretrain \
  -e=16 \
  -b=10 \
  --num_frames=8 \
  --frames=80 \
  -g=1 \
  -se=2 \
  -pt=0.3 \
  -dt=1.1 \
  -d=0.8 \
  -m='./model' \
  -c3d='./conv3d_deepnetA_sport1m_iter_1900000_TF.model'
