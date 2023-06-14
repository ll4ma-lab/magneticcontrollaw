#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:50:54 2023

@author: tabor
"""

from PIL import Image
import os
import numpy as np
from PyPDF2 import PdfWriter, PdfReader, PdfMerger
folder = '/home/tabor/Downloads/ijjr_latex/figures'
def convertLatexCropToPIL(image_size,crop_box):
    #latex is {left bottom right top}
    left,bottom,right,top = crop_box
    width,height = image_size
    
    new_crop = (left,
                top,
                width-right,
                height-bottom)
    return new_crop

def convertLatexCropToPDF(image_size,crop_box):
    #latex is {left bottom right top}
    left,bottom,right,top = crop_box
    width,height = image_size
    
    new_crop = (left,
                height-top,
                width-right,
                bottom)
    return new_crop

redo_characteristic = False
if redo_characteristic:
    sub_folder = '/Characteristic_Trials/'
    output_folder = folder + '_updated' + sub_folder
    try:
        os.makedirs(output_folder)
    except:
        pass
    
    crop_box = (350,50,580,50) # left, top, right, bottom
    
    
        
    file_names = ['cuboid_U_adaptive.jpg',
                  'sphere_prior.jpg',
                  'cuboid.jpg',
                  '8020.jpg',
                  'omni_poor_rotation.jpg',
                  'omni_better_rotation.jpg',
                  'box_U_Adaptive_Strong_init.jpg']
    
    for file_name in file_names:
        im = Image.open(folder + sub_folder + file_name)
        im = im.rotate(1.45)
        PIL_crop = convertLatexCropToPIL(im.size, crop_box)
        im = im.crop(PIL_crop)
        im.save(output_folder+file_name, quality=100)

redo_objects = False
if redo_objects:
    sub_folder = '/objects/'
    output_folder = folder + '_updated' + sub_folder
    try:
        os.makedirs(output_folder)
    except:
        pass
    crop_jobs = [('aluminum_sphere.jpg',(45,15,24,15)),
                 ('copper_sphere.jpg',(45,15,24,15)),
                 ('copper_cuboid.jpg',(29,7,23,5)),
                 ('aluminum_cylinder.jpg',(33,10,32,15)),
                 ('copper_scrap.jpg',(32,10,17,0)),
                 ('omni_magnet.jpg',(15.5,0,4.5,0)),
                 ('80_20.jpg',(27,2,22,8)),
                 ('painted_aluminum_box.jpg',(18,10,12,59)),
                 ('painted_12g_iron.jpg',(25,21,20,64)),
                 ('painted_40g_iron.jpg',(22,30,23,55))]
    for file_name,crop_box in crop_jobs:
        multiplier = 25.4
        crop_box = (np.array(crop_box)*multiplier).tolist()
        im = Image.open(folder + sub_folder + file_name)
        PIL_crop = convertLatexCropToPIL(im.size, crop_box)
        #print(file_name,im.size[0]/im.size[1],im.size)
        im = im.crop(PIL_crop)
        #print(file_name,im.size[0]/im.size[1],im.size)

        #im.show()
        im.save(output_folder+file_name, quality=50)

redo_confidence = False
if redo_confidence:
    sub_folder = '/confidence_plots/'
    output_folder = folder + '_updated' + sub_folder
    try:
        os.makedirs(output_folder)
    except:
        pass
    
    file_names = ['paths_sphere_no_prior',
                  'paths_sphere_prior',
                  'paths_sphere_inv',
                  'paths_aluminum_sphere_no_prior',
                  'paths_aluminum_sphere_prior',
                  'paths_aluminum_sphere_inv',
                  'paths_cuboid',
                  'paths_cylinder',
                  'Paths_pile2',
                  'paths_omni_only',
                  'paths_8020',
                  'paths_box',
                  'Paths_box_small_iron',
                  'Paths_box_big_iron',
                  'paths_pile_easy'
                 ]
    for file_name in file_names:
        file_name = 'P' + file_name[1:] + '.pdf'
        crop_box = (73,63,59,63) # left, top, right, bottom
        pdf = PdfReader(folder + sub_folder + file_name).pages[0]
        
        multiplier = 1
        crop_box = (np.array(crop_box)*multiplier).tolist()
        
        pdf_size = (pdf.mediaBox.width,pdf.mediaBox.height)
        PIL_crop = convertLatexCropToPIL(pdf_size, crop_box)
    
        pdf.cropbox.upper_left = PIL_crop[:2]
        pdf.cropbox.lower_right = PIL_crop[2:]
        writer = PdfWriter()
    
        writer.add_page(pdf) 
        with open(output_folder+file_name,'wb') as fp:
            writer.write(fp) 

redo_sysid_heat = True
if redo_sysid_heat:
    sub_folder = '/sysid/'
    output_folder = folder + '_updated' + sub_folder
    try:
        os.makedirs(output_folder)
    except:
        pass
    
    crop_tasks = [
        (['drag/0.pdf','local_no_prior/0.pdf','local_prior/0.pdf'],(10, 25, 183, 65)),
        (['drag/4.pdf','local_no_prior/4.pdf','local_prior/3.pdf'],(90, 25, 183, 65)),
        (['drag/10.pdf','local_no_prior/10.pdf','local_prior/5.pdf'],(90, 25, 183, 65)),
        (['drag/30.pdf','local_no_prior/23.pdf','local_prior/11.pdf'],(90, 25, 50, 65)),
                 ]
    for file_names,crop_box in crop_tasks:
        for file_name in file_names:
            pdf = PdfReader(folder + sub_folder + file_name).pages[0]
            
    
            
            pdf_size = (pdf.mediaBox.width,pdf.mediaBox.height)
            PIL_crop = convertLatexCropToPDF(pdf_size, crop_box)
        
            pdf.cropbox.setUpperLeft(PIL_crop[:2])
            pdf.cropbox.setLowerRight(PIL_crop[2:])
            print(pdf.cropbox)
            writer = PdfWriter()
        
            writer.add_page(pdf) 
            
            output_file = output_folder + file_name.replace('/','-')
            
            with open(output_file,'wb') as fp:
                writer.write(fp) 



    
