import SimpleITK as sitk
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from tqdm import tqdm
from os.path import join
import random

from skimage.morphology import skeletonize, medial_axis, binary_closing
from scipy.ndimage.morphology import distance_transform_edt

def CC_thickness_profile(iImage, direction):
    
    def resample_image(input_image, spacing_mm=(1, 1, 1), spacing_image=None, inter_type=sitk.sitkNearestNeighbor):
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(inter_type)

        if (spacing_mm is None and spacing_image is None) or            (spacing_mm is not None and spacing_image is not None):
            raise ValueError('You must specify either spacing_mm or spacing_image, not both at the same time.')

        if spacing_image is not None:
            spacing_mm = spacing_image.GetSpacing()

        input_spacing = input_image.GetSpacing()
        # set desired spacing
        resampler.SetOutputSpacing(spacing_mm)
        # compute and set output size
        output_size = np.array(input_image.GetSize()) * np.array(input_spacing)                       / np.array(spacing_mm)
        output_size = list((output_size + 0.5).astype('uint32'))
        output_size = [int(size) for size in output_size]
        resampler.SetSize(output_size)

        resampler.SetOutputOrigin(input_image.GetOrigin())
        resampler.SetOutputDirection(input_image.GetDirection())

        resampled_image = resampler.Execute(input_image)

        return resampled_image
    
    # input sitk Image resamplamo na sampling 1mmx1mm
    iImage_sitk = resample_image(iImage, (1,1,1)) 
    iImage_np = np.squeeze(sitk.GetArrayFromImage(iImage_sitk))
    
    skel = skeletonize(iImage_np)
    dist_tra = distance_transform_edt(iImage_np)

    # Thickness for pixels of the skeleton
    dist_on_skel = dist_tra * skel * 2

    cy, cx = ndimage.center_of_mass(iImage_np)

    print(cx, cy)
    if direction == 'up':
        # polovica cc --> le en krajec
        pol_cc = dist_on_skel[:,:int(cx)]
        # pol tock
        line_pol = np.asarray(np.nonzero(pol_cc))
        # začetna točka kot najvišja točka (oziroma tista z najmanjšo vrstico (v tem primeru y)) 
        tocka0 = line_pol[:,0]
    elif direction == 'down':
        # polovica cc --> le en krajec
        pol_cc = dist_on_skel[:,int(cx):-1]
        # pol tock
        line_pol = np.asarray(np.nonzero(pol_cc))
        tocka0 = line_pol[:,-1]
        tocka0[1] = tocka0[1] + int(cx)

    #print('Začetna točka za cc: ', tocka0)    
    
    i = 0
    debelina = []
    tocka = [tocka0[0],tocka0[1]]
    tocka_history = tocka
    # for zanka za iteracijo premikanja točkah krivulji
    while True:
        flag = True
        # preledovanje okolice piksla (ul, u, ur, r, br, b, bl, l)
        ul = [tocka[0]-1, tocka[1]-1]
        u = [tocka[0]-1, tocka[1]]
        ur = [tocka[0]-1, tocka[1]+1]
        r = [tocka[0], tocka[1]+1]
        br = [tocka[0]+1, tocka[1]+1]
        b = [tocka[0]+1, tocka[1]]
        bl = [tocka[0]+1, tocka[1]-1]
        l = [tocka[0], tocka[1]-1]
        
        # pomemben vrstni red!!! --> če vhodna slika zavita navzgor in želimo iskati po notranjem robu (tam ponavadi ni 
        # bifurkacij skeleta) potem začnemo z iskanjem desno, za navzdol obrnjen CC pa začnemo desno spodaj
        # začetno točko želimo vedno postaviti v splenium
                   
        if direction == 'down':
            #br
            if skel[br[0], br[1]]:
                if br != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = br   
            #b
            if skel[b[0], b[1]]:
                if b != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = b  
            #bl
            if skel[bl[0], bl[1]]:
                if bl != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = bl   
            #l
            if skel[l[0], l[1]]:
                if l != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = l 
            #ul
            if skel[ul[0], ul[1]]:
                if ul != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = ul   
            #u
            if skel[u[0], u[1]]:
                if u != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = u     
            #ur
            if skel[ur[0], ur[1]]:
                if ur != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = ur    
            #r
            if skel[r[0], r[1]]:
                if r != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = r    
        #r
        elif direction == 'up':
            if skel[r[0], r[1]]:
                if r != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = r    

            #br
            if skel[br[0], br[1]]:
                if br != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = br   
            #b
            if skel[b[0], b[1]]:
                if b != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = b  
            #bl
            if skel[bl[0], bl[1]]:
                if bl != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = bl   
            #l
            if skel[l[0], l[1]]:
                if l != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = l 
            #ul
            if skel[ul[0], ul[1]]:
                if ul != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = ul   
            #u
            if skel[u[0], u[1]]:
                if u != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = u    
            #ur
            if skel[ur[0], ur[1]]:
                if ur != tocka_history and flag:
                    flag = False
                    tocka_history = tocka
                    tocka = ur    
        i = i+1
        if flag:
            break
        debelina.append(dist_on_skel[tocka[0], tocka[1]])
        
    debelina = np.asarray(debelina)
    
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(dist_on_skel)
    ax[0].contour(iImage_np, [0.5], colors='w')
    ax[0].scatter(tocka0[1],  tocka0[0], color = 'b', label='Zacetna tocka')
    ax[0].scatter([tocka[1]], [tocka[0]], color = 'r', label='Koncna tocka')
    ax[0].legend()
    ax[1].imshow(pol_cc)
    if direction == 'up':
        ax[1].scatter(tocka0[1], tocka0[0], color = 'b')
    if direction == 'down':        
        ax[1].scatter(tocka0[1]-int(cx), tocka0[0], color = 'b')
    plt.show()  
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(debelina)
    ax.set_ylabel('Debelina v mm')
    ax.set_xlabel('Lokacija na skeletu cc')
    plt.show()
    
    return debelina