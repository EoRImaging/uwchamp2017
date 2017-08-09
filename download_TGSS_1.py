#!/usr/bin/env python
'''A script to download TGSS postage stamps - needs the text file pointing_coversion.txt to work'''
from numpy import loadtxt,cos,sin,pi,arccos
import argparse
from subprocess import call


parser = argparse.ArgumentParser()
parser.add_argument('--coord_list', help='List of target coordinates as RA (deg) Dec (deg)')
parser.add_argument('--im_size', help='Image width (deg) default 0.3 deg',default=0.3,type=float)
args = parser.parse_args()

dr = pi / 180.0

def arcdist(RA1,RA2,Dec1,Dec2):   #calculates distance between two celestial points in degrees
    in1 = (90.0 - Dec1)*dr
    in2 = (90.0 - Dec2)*dr
    RA_d = (RA1 - RA2)*dr
    cosalpha = cos(in1)*cos(in2) + sin(in1)*sin(in2)*cos(RA_d)
    alpha = arccos(cosalpha)
    return alpha/dr

##Input coords to search
ras,decs = loadtxt(args.coord_list,unpack=True)
im_size = args.im_size

print ras, decs

##Need to be able to request the TGSS pointing with field centre
##closest to our target coords - this file holds pointing centres
##and field names
pointing_lines = open('pointing_coversion.txt').read().split('\n')
pointing_label = []
pointing_coords = []

##Create two lists containing coords of field centre and field label
for line in pointing_lines:
    if line == '':
        pass
    else:
        label,RA,DEC = line.split()
        pointing_label.append(label)
        pointing_coords.append([float(RA),float(DEC)])
        
print ras, decs
print zip(ras,decs)
        
for ra,dec in zip(ras,decs):
    ##calculate distance of target location to all field centres
    distances = []
    for p_ra,p_dec in pointing_coords:
        dist = arcdist(p_ra,ra,p_dec,dec)
        distances.append(dist)
    
    ##Find the closest field to target
    pointing = pointing_label[distances.index(min(distances))]
    
    ##Run the wget command using the call command - name the output file based on the ra,dec (the -O option in the string below)
    cmd = 'wget "http://vo.astron.nl/getproduct/tgssadr/fits/TGSSADR_%s_5x5.MOSAIC.FITS?sdec=%.2f&dec=%.10f&ra=%.10f&sra=%.2f" -O downloadTGSS/RA%.2fDEC%.2f_TGSS.fits' %(pointing,im_size,dec,ra,im_size,ra,dec)
    call(cmd,shell=True)
