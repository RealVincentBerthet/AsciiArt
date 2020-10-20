#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""asciiArt.py: AsciiArt program"""
__author__      = 'Vincent Berthet'
__license__     = 'MIT'
__email__       = 'vincent.berthet42@gmail.com'

import cv2 as cv
import numpy as np
import argparse
import os
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, default='me.jpg', help='Path to input image')
    parser.add_argument('-o','--output', type=str, default='', help='Path to output text file')
    parser.add_argument('-r','--rows', type=int, default=256, help='Width of the ouput image')
    parser.add_argument('-c','--columns', type=int, default=256, help='Height of the ouput image')
    parser.add_argument('-d','--dictionnary', type=str, default='simple', choices=['simple', 'complex'],help='Choose dictionanry to use')
    args = parser.parse_args()
    return args

def main(opt):
    # Choose dictionnary
    if opt.dictionnary =='complex' :
        dictionnary = '$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`\'. '
    else:
        dictionnary = '@%#*+=-:. '
        
    # Set output
    (file, ext) = os.path.splitext(os.path.basename(opt.input))
    if opt.output=='' :
        path=file+'-'+datetime.now().strftime("%Y-%m-%d_%H%M%S")+'.txt'
    else:
        path=opt.output

    # Process ASCIIArt
    src = cv.imread(opt.input)
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src = cv.resize(src,(opt.rows,int(opt.columns/2)),interpolation=cv.INTER_CUBIC)

    output_file = open(path, 'w')
    for j in range(src.shape[0]):
        for i in range(src.shape[1]):
            index=int(src[j,i]*len(dictionnary)/255)
            output_file.write(dictionnary[index])
        output_file.write("\n")
    output_file.close()

if __name__ == '__main__':
    opt = get_args()
    main(opt)