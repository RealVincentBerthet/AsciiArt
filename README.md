# ASCII Art

[![License](https://img.shields.io/github/license/RealVincentBerthet/ASCIIArt?style=flat-square)](https://opensource.org/licenses/MIT)

## Abstract
Process an image to create ASCII Art. 

Output can be a text file (.txt) or an image (.jpg, .png). You can tune the different optionnal parameters of [scripts](#Scripts) to change the output result.


| ![](./output/sample-2020-10-20_164055.jpg) | ![](./output/sample-2020-10-20_164058.jpg) | ![](./output/sample-2020-10-20_164059.jpg) | 
|:-:|:-:|:-:| 
| ![](./output/sample-2020-10-20_164100.jpg) | ![](./output/sample-2020-10-20_164054.jpg) | ![](./output/sample-2020-10-20_182818.jpg) | 
| ![](./output/sample-2020-10-20_183120.jpg) | ![](./output/sample-2020-10-20_183600.jpg) | ![](./output/sample-2020-10-20_214012.jpg) | 


## Packages
*Python 3.8*

[OpenCV 4.4.0](https://anaconda.org/conda-forge/opencv)
`conda install -c conda-forge opencv=4.4.0`

[Pillow 8.0.0](https://anaconda.org/conda-forge/pillow)
`conda install -c conda-forge pillow=8.0.0 `

## Scripts
### Description
* [asciiArt.py](./asciiArt.py), Main script to generate ASCII Art from a source image. You can override default parameters with the following argurments.

    * `-i, --input` : Path to the input image.
    * `-o, --output` : Path to the output file
    * `-r, --rows` : Number of ASCII rows
    * `-c, --columns` : Number of ASCII columns
    * `-d, --dictionnary` : Choose a dictionnary to use (simple, complex, vberthet)
    * `-f, --font` : [IMG only] Font to use
    * `-fs, --fontSize` : [IMG only] Size of the font
    * `-bg, --background` : [IMG only] Background color
    * `--debug` :  Enable debug


### Run 
**TXT :** `python asciiArt.py -i sample.jpg -o sample.txt -r 256 -c 256 -d vberthet --debug`

**IMG :** `python asciiArt.py -i sample.jpg -o sample.png -r 256 -c 256 -d vberthet -f fonts/deja-vu/DejaVuSansMono-Bold.ttf -fs 8 -bg 255 --debug`