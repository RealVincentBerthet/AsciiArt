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
from PIL import Image, ImageFont, ImageDraw, ImageOps
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, default='sample.jpg', help='Path to input image')
    parser.add_argument('-o','--output', type=str, default=' .jpg', help='Path to output file')
    parser.add_argument('-r','--rows', type=int, default=256, help='Number of ASCII rows')
    parser.add_argument('-c','--columns', type=int, default=256, help='Number of ASCII columns')
    parser.add_argument('-d','--dictionnary', type=str, default='simple', choices=['simple', 'complex', 'vberthet'], help='Choose dictionnary to use')
    parser.add_argument('-f','--font', type=str, default='fonts/deja-vu/DejaVuSansMono-Bold.ttf', choices=['fonts/deja-vu/DejaVuSansMono-Bold.ttf','fonts/caviar-dreams/Caviar Dreams Bold.ttf'], help='[IMG only] Font to use')
    parser.add_argument('-fs','--fontSize',type=int,default=10,help='[IMG only] Size of the font')
    parser.add_argument('-bg','--background', type=int, default=255, help='[IMG only] Background color')
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    return args

def toTxt(img,dictionnary,path):
    output_file = open(path, 'w')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index=int(img[i,j]*len(dictionnary)/255)
            output_file.write(dictionnary[index])
        output_file.write("\n")
    output_file.close()

    if opt.debug:
        with open(path, 'r') as f:
            print(f.read())

    print('[TXT]',path,'saved')

def toImg(img,dictionnary,path):
    font = ImageFont.truetype(opt.font, size=opt.fontSize)
    font_width, font_height = font.getsize('X')
    out_width = font_width * opt.columns
    out_height = int(font_height * opt.rows/2)
    output_file = Image.new('L', (out_width, out_height), opt.background) # (8-bit pixels, black and white)
    draw = ImageDraw.Draw(output_file)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index=int(img[i,j]*len(dictionnary)/255)
            draw.text((j*font_width, i * font_height), dictionnary[index], fill=255 - opt.background, font=font)

    if opt.debug :
        output_file.show()

    output_file.save(path)
    print('[IMG]',path,'saved')

def main(opt):
    # Choose dictionnary
    if opt.dictionnary =='complex' :
        dictionnary = '$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`\'. '
    elif opt.dictionnary =='vberthet' :
        dictionnary = 'VBERTHET'
    else:
        dictionnary = '@%#*+=-:. '
        
    # Set output
    (file, ext) = os.path.splitext(os.path.basename(opt.output))
    if file=='' or ' '  : 
        path='output/'+os.path.splitext(os.path.basename(opt.input))[0]+'-'+datetime.now().strftime("%Y-%m-%d_%H%M%S")+ext
    else:
        path=opt.output

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    # Pre-process input image
    src = cv.imread(opt.input)
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src = cv.resize(src,(opt.columns,int(opt.rows/2)),interpolation=cv.INTER_CUBIC)

    if ext=='.txt':
        toTxt(src,dictionnary,path)
    else:
        toImg(src,dictionnary,path)

if __name__ == '__main__':
    opt = get_args()
    main(opt)