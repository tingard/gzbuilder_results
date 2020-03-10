import os
import sys
from tqdm import tqdm
from time import time
import subprocess
import argparse


def convert(infile, outfolder=None, density=300, resize='25%'):
    assert infile.split('.')[-1] == 'pdf'
    infolder, infile_ = os.path.split(infile)
    if outfolder is None:
        outfolder = infolder
    os.makedirs(outfolder, exist_ok=True)
    outfile = os.path.abspath(os.path.join(
        outfolder,
        '.'.join(infile_.split('.')[:-1] + ['png'])
    ))
    s = f'convert -density {density} {infile} -resize {resize}% {outfile}'
    return subprocess.call(s, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert from a pdf to a png'
    )
    parser.add_argument('infile', metavar='/path/to/file.pdf', nargs='*',
                        help='PDF image to convert (file path)')
    parser.add_argument('--outfolder', metavar='/path/to/output/folder/',
                        help='Output folder')
    parser.add_argument('--resize', '-r', metavar='N%', default='25%',
                        help='Output image size')
    parser.add_argument('--density', '-d', metavar='M', default=300,
                        help='output image density (number)')
    args = parser.parse_args()
    t0 = time()
    with tqdm(args.infile, desc='Converting files') as bar:
        for i, f in enumerate(bar):
            try:
                convert(f, args.outfolder, args.density, args.resize)
            except AssertionError:
                pass
        print('\nCompleted in {:.3f}s'.format(time() - t0))
