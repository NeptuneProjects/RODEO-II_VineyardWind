#!/bin/bash

3DVHA_DIR="./data/acoustic/raw/3DVHA/acoustics"
cp sftp://castilon//project/users/ytlin/VW_DAS/3DVHA/acoustics/20231201/ *.bin $3DVHA_DIR

VLA1_DIR="./data/acoustic/raw/VLA1_shru905/"
cp /project/users/ytlin/VW_DAS/SHRUs/VLA1_shru905/12011930.D23 $VLA1_DIR
cp /project/users/ytlin/VW_DAS/SHRUs/VLA1_shru905/12012125.D23 $VLA1_DIR
cp /project/users/ytlin/VW_DAS/SHRUs/VLA1_shru905/12012319.D23 $VLA1_DIR

VLA2_DIR="./data/acoustic/ra/VLA2_shru92/"
cp /project/users/ytlin/VW_DAS/SHRUs/VLA2_shru920/12011948.D23 $VLA2_DIR
cp /project/users/ytlin/VW_DAS/SHRUs/VLA2_shru920/12012143.D23 $VLA2_DIR
cp /project/users/ytlin/VW_DAS/SHRUs/VLA2_shru920/12012338.D23 $VLA2_DIR
