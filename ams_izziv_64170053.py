import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
from tqdm import tqdm
from os.path import join

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from scipy import ndimage
from scipy.interpolate import interpn    
from scipy.ndimage import label

from sklearn.model_selection import train_test_split

import skimage.morphology as morph

from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
from skimage import exposure
def midsaggital_plane_extraction(t1_image):
    
    orig_orient = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(t1_image.GetDirection())
    orientation_filter = sitk.DICOMOrientImageFilter()
    if orig_orient is not 'PSR':
        t1 = sitk.DICOMOrient(t1_image, 'PSR')
        
    o = t1.GetOrigin()
    s = t1.GetSpacing()
    d = t1.GetDirection()

    # normalize image
    t11 = sitk.GetArrayFromImage(t1)
    t11 = (t11 - (np.min(t11)-1e-12)) / (np.max(t11) - (np.min(t11)-1e-12))

    # reshape the 3d arrays such that the number of cases is in the first column and then y, x, z
    # x,z,y -> y,x,z
    t1_array = np.transpose(t11, (2, 0, 1))
    im = np.array(t1_array, dtype = 'float64')
    
    H, W, D = im.shape
    cy, cx, cz = ndimage.center_of_mass(im)
        
    ########################################################
    # point definiton that defines mid-sagittal plane
    T1 = np.array([cy, cx, cz])
    T2 = np.array([cy + 10, cx, cz])
    T3 = np.array([cy, cx, cz - 10])
    
    #n = (T2−T1)×(T3−T1) -> normal vector
    n = np.cross(T2-T1, T3-T1)
    n = n/np.linalg.norm(n)

    #Plane equation: n = (b,a,c) --> ax+by+cz+k=0
    k = -n[0]*T1[0] -n[1]*T1[1] -n[2]*T1[2]

    # Mesh for mid_sagittal plane definition
    yy, zz = np.meshgrid(range(H),(range(D)))
    # calculate plane
    xx = (-n[2]*zz - n[0]*yy - k)/n[1] 

    # interpolate 3D image in plane coordinates
    iI = interpn((np.arange(im.shape[0]),np.arange(im.shape[1]),np.arange(im.shape[2])),
                     im.astype('float64'),
                     (yy,xx,zz), 
                     method = 'linear',
                     fill_value = 0,
                     bounds_error = False)
    
    # add additional dimension
    oI = sitk.JoinSeries(sitk.GetImageFromArray(iI))
    
    # set parameters
    oI.SetSpacing(s)
    oI.SetDirection(d)
    oo = np.array((0,o[1],o[2]))
    oI.SetOrigin(oo)
    
    # reorient to original orientation
    if orig_orient is not 'PSR':
        orientation_filter = sitk.DICOMOrientImageFilter()
        oI = sitk.DICOMOrient(oI, orig_orient)
    
    t1_slice = oI
    
    return t1_slice


def corpus_callosum_segmentation(t1_slice):
    #############################################
    ########## arhitecture of models ############
    #############################################
    
    # Model 0
    h0 = 64
    w0 = 64
    inputs = Input((h0,w0, 1))

    s = Lambda(lambda x: x)(inputs)

    # Levi del U
    c1 = Conv2D(32, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(s) 
    c1 = Conv2D(32, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c1) 
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2,2))(c1)

    c2 = Conv2D(64, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(p1)
    c2 = Conv2D(64, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2,2))(c2)

    c3 = Conv2D(128, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(p2)
    c3 = Conv2D(128, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2,2))(c3)

    c4 = Conv2D(256, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(p3)
    c4 = Conv2D(256, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c4)
    c4 = BatchNormalization()(c4)

    u7 = Conv2DTranspose(128, (2,2), strides = (2,2), padding = 'same')(c4)
    u7 = concatenate([u7,c3])
    c7 = Dropout(0.2)(u7) 
    c7 = Conv2D(128, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c7)
    c7 = Conv2D(128, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c7)

    u8 = Conv2DTranspose(64, (2,2), strides = (2,2), padding = 'same')(c7)
    u8 = concatenate([u8,c2])
    c8 = Dropout(0.2)(u8) 
    c8 = Conv2D(64, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c8)
    c8 = Conv2D(64, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c8)

    u9 = Conv2DTranspose(32, (2,2), strides = (2,2), padding = 'same')(c8)
    u9 = concatenate([u9,c1], axis = 3) 
    c9 = Dropout(0.2)(u9) 
    c9 = Conv2D(32, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c9)
    c9 = Conv2D(32, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c9)

    outputs = Conv2D(1, (1,1), activation = 'sigmoid')(c9)

    model0 = Model(inputs = [inputs], outputs = [outputs])
    
    
    # Model 1    
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 2
    # vhodna plast
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    s = Lambda(lambda x: x)(inputs)

    # Levi del U
    c1 = Conv2D(32, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(s) 
    c1 = Conv2D(32, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c1) 
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2,2))(c1)

    c2 = Conv2D(64, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(p1)
    c2 = Conv2D(64, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2,2))(c2)

    c3 = Conv2D(128, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(p2)
    c3 = Conv2D(128, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2,2))(c3)

    c4 = Conv2D(256, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(p3)
    c4 = Conv2D(256, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D((2,2))(c4)

    c41 = Conv2D(512, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(p4)
    c41 = Conv2D(512, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c41)
    c41 = BatchNormalization()(c41)

    p41 = MaxPooling2D((2,2))(c41)

    c5 = Conv2D(1024, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(p41)
    c5 = Conv2D(1024, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c5)
    c5 = BatchNormalization()(c5)
    # desni del U
    u61 = Conv2DTranspose(512, (2,2), strides = (2,2), padding = 'same')(c5)
    u61 = concatenate([u61,c41])
    c61 = Dropout(0.2)(u61) 
    c61 = Conv2D(512, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c61)
    c61 = Conv2D(512, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c61)

    u6 = Conv2DTranspose(256, (2,2), strides = (2,2), padding = 'same')(c61)
    u6 = concatenate([u6,c4])
    c6 = Dropout(0.2)(u6) 
    c6 = Conv2D(256, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c6)
    c6 = Conv2D(256, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c6)

    u7 = Conv2DTranspose(128, (2,2), strides = (2,2), padding = 'same')(c6)
    u7 = concatenate([u7,c3])
    c7 = Dropout(0.2)(u7) 
    c7 = Conv2D(128, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c7)
    c7 = Conv2D(128, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c7)

    u8 = Conv2DTranspose(64, (2,2), strides = (2,2), padding = 'same')(c7)
    u8 = concatenate([u8,c2])
    c8 = Dropout(0.2)(u8) 
    c8 = Conv2D(64, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c8)
    c8 = Conv2D(64, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c8)

    u9 = Conv2DTranspose(32, (2,2), strides = (2,2), padding = 'same')(c8)
    u9 = concatenate([u9,c1], axis = 3) 
    c9 = Dropout(0.2)(u9) 
    c9 = Conv2D(32, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c9)
    c9 = Conv2D(32, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(c9)

    outputs = Conv2D(1, (1,1), activation = 'sigmoid')(c9)

    model1 = Model(inputs = [inputs], outputs = [outputs])

    
    #########################################################################################################
    #################################### Functions ##########################################################
    #########################################################################################################
    
    
    def dice_coef(y_true, y_pred):
        """
        DSC = (2*|X &amp; Y|)/ (|X| + |Y|)
        """    
        smooth = 1
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return 1-dice_coef(y_true, y_pred)
    
    def resample_image123(itk_image, out_size=(1, 256, 256), is_label=False, s = None):
        original_spacing = itk_image.GetSpacing()
        original_size = itk_image.GetSize()
        if s is None:
            out_spacing = [original_size[0] * (original_spacing[0]) / out_size[0],
                           original_size[1] * (original_spacing[1]) / out_size[1],
                           original_size[2] * (original_spacing[2]) / out_size[2]
                          ]
        else:
            out_spacing = [s[0], s[1], s[2]]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)

        return resample.Execute(itk_image)
    
    def resample_image(input_image, spacing_mm=(1, 1, 1), spacing_image=None, inter_type=sitk.sitkLinear):
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(inter_type)

        if (spacing_mm is None and spacing_image is None) or \
           (spacing_mm is not None and spacing_image is not None):
            raise ValueError('You must specify either spacing_mm or spacing_image, not both at the same time.')

        if spacing_image is not None:
            spacing_mm = spacing_image.GetSpacing()

        input_spacing = input_image.GetSpacing()
        # set desired spacing
        resampler.SetOutputSpacing(spacing_mm)
        # compute and set output size
        output_size = np.array(input_image.GetSize()) * np.array(input_spacing) \
                      / np.array(spacing_mm)
        output_size = list((output_size + 0.5).astype('uint32'))
        output_size = [int(size) for size in output_size]
        resampler.SetSize(output_size)

        resampler.SetOutputOrigin(input_image.GetOrigin())
        resampler.SetOutputDirection(input_image.GetDirection())

        resampled_image = resampler.Execute(input_image)

        return resampled_image
    
    def reshape_set(iIm_set, out_shape = (64,64), int_type = sitk.sitkLinear):
        def extract(image, output_size=(128, 128), interpolation_type=sitk.sitkLinear):
            H, W= image.GetSize()
            H_s, W_s = image.GetSpacing()
            new_spacing_mm = (H / output_size[0] * H_s, W / output_size[1] * W_s)
            return resample_image(image, 
                spacing_mm = new_spacing_mm, 
                inter_type=interpolation_type)
    
        oIm_set = np.zeros((iIm_set.shape[0],out_shape[0],out_shape[1],1))
        for i, img in enumerate(iIm_set):
            x1 = sitk.GetArrayFromImage(
                extract(sitk.GetImageFromArray(img[:,:,0]), 
                                output_size=(out_shape[0],out_shape[1]),
                                interpolation_type=int_type))
            oIm_set[i,:,:,0] = x1
        return oIm_set
    
    # meta parameters save
    original_size = t1_slice.GetSize()
    o = t1_slice.GetOrigin()
    d = t1_slice.GetDirection()
    s = t1_slice.GetSpacing()
    
    # reorient if necessary
    orig_orient = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(t1_slice.GetDirection())
    orientation_filter = sitk.DICOMOrientImageFilter()
    if orig_orient is not 'LPS':
        t1_slice = sitk.DICOMOrient(t1_slice, 'LPS')
        
    # poskusi zakomentirati tazgorno
    original_size2 = t1_slice.GetSize()
    
    # reshape into 1x256X256
    rezina2 = resample_image123(t1_slice, out_size=[1, 256, 256], is_label=False)
    rezina_np = sitk.GetArrayFromImage(rezina2)[np.newaxis,:,:,:]
        
    ########################## PREPROCCESSING ##############################################################
    # normalizacija slike na [0-1]
    rezina_np = (rezina_np - (np.min(rezina_np) - 1e-12)) / (np.max(rezina_np) - (np.min(rezina_np) - 1e-12))
    # linearno oknjenje    
    p_l, p_u = np.percentile(rezina_np, (45, 99))
    rezina_np = exposure.rescale_intensity(rezina_np, in_range=(p_l, p_u))
    
    #########################################################################################################
    # predikcije modelov
    # reshape slik v 64x64
    d0 = reshape_set(rezina_np, out_shape = (64,64))
    
    # prvi model
    model0 = load_model('model0-corpuss_callosum8_swirl5.h5', 
                   custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})
    rezina_pred0 = model0.predict(d0)
    rezina_pred_t0 = (rezina_pred0 > 0.0).astype(np.uint8)
        
    # reshape maske iz 64x64 v 256x256 in concatenate z rezina_np v 3 dimenzijo
    pr0 = reshape_set(rezina_pred_t0, out_shape = (256,256), int_type = sitk.sitkNearestNeighbor)
    d1 = np.concatenate((rezina_np, pr0), axis = 3)    
    
    # drugi model
    model1 = load_model('model1-corpuss_callosum8_swirl5.h5', 
                   custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})
    rezina_pred = model1.predict(d1)
    rezina_pred_t = (rezina_pred > 0.0).astype(np.uint8)
    
    #########################################################################################################
    # Postprocessing
    
    # deleting objects smaller than 0.3 of largest connected component    
    segmentation_mask = rezina_pred_t[0,:,:,0]
    binary_mask = segmentation_mask.copy()
    binary_mask[binary_mask != 0] = 1
    labelled_mask, num_labels = label(binary_mask)    
    refined_mask = segmentation_mask.copy()
    
    minimum_cc_sum = 0
    for lab1 in range(num_labels):
        minimum_cc_sum_tmp = np.sum(refined_mask[labelled_mask == (lab1+1)])
        minimum_cc_sum = np.maximum(minimum_cc_sum, minimum_cc_sum_tmp)
    minimum_cc_sum = 0.3 * minimum_cc_sum
    
    for lab in range(num_labels):
        #print(np.sum(refined_mask[labelled_mask == (lab+1)]))        
        if np.sum(refined_mask[labelled_mask == (lab+1)]) < minimum_cc_sum:
            #print('Izbris, ker število pikslov: ', np.sum(refined_mask[labelled_mask == (lab+1)]))
            refined_mask[labelled_mask == (lab+1)] = 0
    
    # morfološko zapiranje
    refined_mask = morph.closing(refined_mask)
    
    #########################################################################################################
    
    # Pretvarjanje maske iz Array v SitkImage
    mask_sitk = sitk.GetImageFromArray(refined_mask[:, :, np.newaxis])
    
    # Sprememba dimenzij, orientacije in parametrov v originalne  
    rezina_pred_t2 = resample_image123(mask_sitk, original_size2, is_label=True)
    
    orientation_filter = sitk.DICOMOrientImageFilter()
    if orig_orient is not 'LPS':
        rezina_pred_t2 = sitk.DICOMOrient(rezina_pred_t2, orig_orient)
        
    rezina_pred_t2.SetDirection(d)
    rezina_pred_t2.SetOrigin(o)
    rezina_pred_t2.SetSpacing(s)
    
    cc_seg = rezina_pred_t2
    
    return cc_seg


def midsaggital_plane_extraction_ellipse(t1_image):
    def fit_ellipse(x, y):
        x = x[ :, np.newaxis ]
        y = y[ :, np.newaxis ]
        D =  np.hstack( ( x * x, x * y, y * y, x, y, np.ones_like( x ) ) )
        S = np.dot( D.T, D )
        C = np.zeros( [ 6, 6 ] )
        C[ 0, 2 ] = +2 
        C[ 2, 0 ] = +2
        C[ 1, 1 ] = -1
        E, V =  np.linalg.eig( np.dot( np.linalg.inv( S ), C ) )
        n = np.argmax( np.abs( E ) )
        a = V[ :, n ]
        return a

    def ell_parameters( a ):
        A = np.array( [ [ a[0], a[1]/2. ], [ a[1]/2., a[2] ] ] )
        b = np.array( [ a[3], a[4] ] )
        t = np.dot( np.linalg.inv( np.transpose( A ) + A ), b )
        c = a[5]
        cnew =  c - np.dot( t, b ) + np.dot( t, np.dot( A, t ) )
        Anew = A / (-cnew)
        E, V = np.linalg.eig( Anew )
        phi = np.arccos( V[ 0, 0 ] )
        if V[ 0, 1 ] < 0: 
            phi = 2 * np.pi - phi
        if phi > np.pi:
            phi = phi - np.pi
        phi = -phi % np.pi
        return np.sqrt( 1. / E ), phi * 180. / np.pi, -t
    
    orig_orient = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(t1_image.GetDirection())
    orientation_filter = sitk.DICOMOrientImageFilter()
    if orig_orient is not 'PSR':
        t1 = sitk.DICOMOrient(t1_image, 'PSR')
        
    o = t1.GetOrigin()
    s = t1.GetSpacing()
    d = t1.GetDirection()

    # normalize image
    t11 = sitk.GetArrayFromImage(t1)
    t11 = (t11 - (np.min(t11)-1e-12)) / (np.max(t11) - (np.min(t11)-1e-12))

    # reshape the 3d arrays such that the number of cases is in the first column and then y, x, z
    # x,z,y -> y,x,z
    t1_array = np.transpose(t11, (2, 0, 1))
    im = np.array(t1_array, dtype = 'float64')
    
    H, W, D = im.shape
    cy, cx, cz = ndimage.center_of_mass(im)

    center_axial = []
    center_coronal = []
    phi_axial = []
    phi_coronal = []
    # y,x,z
    #finding centers in axial cross section --> only one plane works best
    no_planes = 0
    for i in range(int(cz)-no_planes//2,int(cz)+no_planes//2 + 1):
        y, x = np.nonzero(im[:,:,i])
        if x.size and y.size:
            aa = fit_ellipse(x,y)
            (a, b), phi, t = ell_parameters(aa)
            center_axial.append(t)
            phi_axial.append(phi)
    center_axial = np.array(center_axial)
    phi_axial = np.array(phi_axial)

    # v coronalni ravnini prilegam elipso na rezini, ki ustreza y središču axialne elipse
    y, x = np.nonzero(im[int(center_axial[0,1]),:,:]) 
    if x.size and y.size:
        aa = fit_ellipse(x,y)
        (a, b), phi, t = ell_parameters(aa)
        center_coronal.append(t)
        phi_coronal.append(phi)
    center_coronal = np.array(center_coronal)
    phi_coronal = np.array(phi_coronal)

    czy = center_axial[0,1]
    czx = center_axial[0,0]

    cyx = center_coronal[0,1]
    cyz = center_coronal[0,0]

    ph = np.deg2rad(np.mean(phi_axial))
    ph2 = np.deg2rad(np.mean(phi_coronal))

    ################################################################
    T1 = np.array([czy, czx, cz])
    T2 = np.array([czy + 10*np.cos(ph), czx + 10*np.sin(ph), cz])
    T31 = np.array([czy, czx, cyz])
    T3 = np.array([czy, czx + 10*np.sin(ph2), cyz + 10*np.cos(ph2)])
    #################################################################

    #n = (B−A)×(C−A) -> normal vector
    n = np.cross(T2-T1, T3-T1)
    n = n/np.linalg.norm(n)
    #Plane equation: n = (b,a,c) --> ax+by+cz+k=0
    k = -n[0]*T1[0] -n[1]*T1[1] -n[2]*T1[2]

    # Mesh for mid_sagittal plane definition
    yy, zz = np.meshgrid(range(H),(range(D)))
    # calculate plane
    xx = (-n[2]*zz - n[0]*yy - k)/n[1] 

    # interpolate 3D image in plane coordinates
    iI = interpn((np.arange(im.shape[0]),np.arange(im.shape[1]),np.arange(im.shape[2])),
                     im.astype('float64'),
                     (yy,xx,zz), 
                     method = 'linear',
                     fill_value = 0,
                     bounds_error = False)
    
    # add additional dimension
    oI = sitk.JoinSeries(sitk.GetImageFromArray(iI))
    
    # set parameters
    oI.SetSpacing(s)
    oI.SetDirection(d)
    oo = np.array((0,o[1],o[2]))
    oI.SetOrigin(oo)
    
    # reorient to original orientation
    if orig_orient is not 'PSR':
        orientation_filter = sitk.DICOMOrientImageFilter()
        oI = sitk.DICOMOrient(oI, orig_orient)
    
    t1_slice = oI

    return t1_slice