#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 20:05:55 2023

@author: tabor
"""

from PyPDF2 import PdfFileWriter, PdfFileReader
from PyPDF2 import PageObject

pdf_filenames = ["/home/tabor/Downloads/ijjr_latex/figures_updated/sysid/drag-0.pdf",
                "/home/tabor/Downloads/ijjr_latex/figures_updated/sysid/drag-4.pdf",
                "/home/tabor/Downloads/ijjr_latex/figures_updated/sysid/drag-10.pdf",
                "/home/tabor/Downloads/ijjr_latex/figures_updated/sysid/drag-30.pdf"]


inputs = [PdfFileReader(open(pdf_filenames, "rb"), strict=False) for pdf_filenames in pdf_filenames]

total_width = sum([pdf.getPage(0).cropbox.width for pdf in inputs])
total_height = max([pdf.getPage(0).cropbox.height for pdf in inputs])

new_page = PageObject.createBlankPage(None, total_width, total_height)

current_x = 0

for pdf in inputs:
    curr_page = pdf.getPage(0)
    curr_cropbox = curr_page.cropbox
    new_page.mergeTranslatedPage(curr_page,current_x-curr_cropbox.left, -curr_cropbox.bottom)
    current_x += curr_cropbox.width 


output = PdfFileWriter()
output.addPage(new_page)
output.write(open("result.pdf", "wb"))