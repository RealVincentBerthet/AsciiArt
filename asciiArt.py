#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""asciiArt.py: AsciiArt program"""
__author__      = 'Vincent Berthet'
__license__     = 'MIT'
__email__       = 'vincent.berthet42@gmail.com'
__website__     = 'https://realvincentberthet.github.io/vberthet/'

import cv2 as cv
import numpy as np
import argparse
import os
from PIL import Image, ImageFont, ImageDraw, ImageOps
from datetime import datetime
from CameraOpenCV import Camera

def get_args():
    """
    get_args function return optional parameters.

    :return: argurments set by default or overriden
    """
    parser = argparse.ArgumentParser(description='Process an image to create ASCII Art. Output can be a text file [.txt] or an image [.jpg, .png]. You can tune the different optionnal parameters to change the result. More details in the dedicated README.')
    parser.add_argument('-i','--input', type=str, default='sample.jpg', help='Path to the input image (can be a video or you camera)')
    parser.add_argument('-o','--output', type=str, default='', help='Path to the output file')
    parser.add_argument('-r','--rows', type=int, default=256, help='Number of ASCII rows')
    parser.add_argument('-c','--columns', type=int, default=256, help='Number of ASCII columns')
    parser.add_argument('-d','--dictionnary', type=str, default='simple', choices=['simple', 'complex', 'vberthet'], help='Choose a dictionnary to use')
    parser.add_argument('-f','--font', type=str, default='fonts/deja-vu/DejaVuSansMono-Bold.ttf', choices=['fonts/deja-vu/DejaVuSansMono-Bold.ttf','fonts/caviar-dreams/CaviarDreams_Bold.ttf'], help='[IMG only] Font to use')
    parser.add_argument('-fs','--fontSize',type=int,default=10,help='[IMG only] Size of the font')
    parser.add_argument('-bg','--background', type=int, default=255, help='[IMG only] Background color')
    parser.add_argument('--debug', default=False, action='store_true', help='Enable debug and show results')
    parser.add_argument('--txt', default=False, action='store_true', help='Force output to .txt file')
    parser.add_argument('--unsave', default=False, action='store_true', help='Force to unsave the output file')

    args = parser.parse_args()
    return args

def preProcess(src):
    # Pre-process input image
    if(isinstance(src, str)):
        img = cv.imread(src)
    else :
        img=src

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img,(opt.columns,int(opt.rows/2)),interpolation=cv.INTER_CUBIC)

    return img

def toTxt(src,dictionnary,path=None):
    """
    toTxt function process an image to create an ASCII Art text file

    :param img: source image in grayscale 8-bits C1
    :param dictionnary: ASCII char list used to encode pixels value
    :param path: path to the output file
    """
    img=preProcess(src)
    text=''
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index=int(img[i,j]*len(dictionnary)/256)
            text+=dictionnary[index]
        text+='\n'
    
    if not opt.unsave and not path==None:
        output_file = open(path, 'w')
        output_file.write(text)
        output_file.close()
        print('[TXT]',path,'saved')

    return text

def toImg(src,dictionnary,path=None):
    """
    toImg function process an image to create an ASCII Art image file

    :param img: source image in grayscale 8-bits C1
    :param dictionnary: ASCII char list used to encode pixels value
    :param path: path to the output file
    """
    img=preProcess(src)

    font = ImageFont.truetype(opt.font, size=opt.fontSize)
    font_width, font_height = font.getsize('X')
    out_width = font_width * opt.columns
    out_height = int(font_height * opt.rows/2)
    output_file = Image.new('L', (out_width, out_height), opt.background) # (8-bit pixels, black and white)
    draw = ImageDraw.Draw(output_file)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index=int(img[i,j]*len(dictionnary)/256)
            draw.text((j*font_width, i * font_height), dictionnary[index], fill=255 - opt.background, font=font)

    if not opt.unsave and not path==None :
        output_file.save(path)
        print('[IMG]',path,'saved')

    return output_file

def main(opt):
    # Choose dictionnary
    if opt.dictionnary =='complex' :
        dictionnary = '$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`\'. '
    elif opt.dictionnary =='vberthet' :
        dictionnary = 'VBERTHET'
    else:
        dictionnary = '@%#*+=-:. '
        
    # Set output
    (ifile, iext) = os.path.splitext(os.path.basename(opt.input))
    (file, ext) = os.path.splitext(os.path.basename(opt.output))
    if file=='' or ext=='' : 
        ext='.txt' if ext=='' and opt.txt else '.jpg'
        path='output/'+os.path.splitext(os.path.basename(opt.input))[0]+'-'+datetime.now().strftime('%Y-%m-%d_%H%M%S')+ext
    else:
        path=opt.output

    if not os.path.dirname(path)=='' and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    if iext=='.jpg' or iext=='.png' or iext=='.jpeg':
        if opt.txt:
            art=toTxt(opt.input,dictionnary,path)
            if opt.debug :
                print(art)
        else:
            art=toImg(opt.input,dictionnary,path)
            if opt.debug :
                art.show()
    else:
        print('[INFO] Use video as the input (video file or streaming from camera)')
        print('[INFO] To help realtime conversion : only image to .txt will be done, save output is disabled')
        camera=Camera(opt.input)
        while True:
            frame=camera.getFrame()
            if frame.all() :
                break
            else:
                cv.imshow('Source',frame)
                cv.waitKey(1)
                art=toTxt(frame,dictionnary)
                print(art)

if __name__ == '__main__':
    opt = get_args()
    main(opt)