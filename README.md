# How to use the stuff

1. Install Python 3, tensorflow 2.0+, python-opencv, numpy, colour-science
2. Prepare a folder of images, PNG or JPG only
3. Run `python tools/dataset.py [your folder of images] [output data folder] [Camera Model]` to create training dataset
   1. This will downscale and crop the images to $128x128$ sections
   2. The image will be converted into binary floating point format
   3. The image will be converted into simulated CFAs
   4. See available camera models in `data/camlist&equipment.txt`
   5. The folder contains the original images SHOULD NOT be the same as the output data folder
4. Run `python train.py [output data folder]` and wishes for the best