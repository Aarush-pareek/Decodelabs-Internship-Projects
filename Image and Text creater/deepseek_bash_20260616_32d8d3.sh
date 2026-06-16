#!/bin/bash
mkdir -p mobilenet_ssd
cd mobilenet_ssd
wget https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt
wget https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc -O mobilenet_iter_73000.caffemodel