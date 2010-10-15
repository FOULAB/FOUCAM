#!/usr/bin/env python

import sys
import pygame
from pygame.locals import *

import opencv 
from opencv import highgui

import time
import signal
from multiprocessing import *

try:
    import psyco
    psyco.full()
except ImportError:
    pass


def FaceDetect( image ):
    storage = opencv.cvCreateMemStorage( 0 )
    opencv.cvClearMemStorage( storage )
    cascade = opencv.cvLoadHaarClassifierCascade( 'haarcascade_frontalface.xml' , opencv.cvSize( 1, 1 ) )
    mugsht = opencv.cvHaarDetectObjects( image,
                                         cascade,
                                         storage,
                                         1.2,
                                         2,
                                         opencv.CV_HAAR_DO_CANNY_PRUNING,
                                         opencv.cvSize( 75, 75 ) )
    if mugsht:
        for mug in mugsht:
            face = [ 0, 0, 0, 0 ]
            face[0], face[1], face[2], face[3] = mug.x, mug.y, mug.width, mug.height 
            faces.append( face )


class GetImage( object ):
    
    def convert_PIL_to_pygame( self, image_PIL ):
        return pygame.image.frombuffer( image_PIL.tostring(), image_PIL.size, image_PIL.mode )
    
    def get_image_Ipl( self, camera ):
        img = highgui.cvQueryFrame( camera )
        FaceDetect( img )
        #opencv.cvSmooth( img, img, opencv.CV_GAUSSIAN, 9, 9 )
        return img
    
    def get_image_PIL( self, camera ):
        return opencv.adaptors.Ipl2PIL( self.get_image_Ipl( camera ) )

    def get_image_pygame( self, camera ):
        return convert_PIL_to_pygame( self.get_image_PIL( camera ) )     


getimage = GetImage()

width = 640
height = 480

camera = opencv.highgui.cvCreateCameraCapture(0)
opencv.highgui.cvSetCaptureProperty( camera, opencv.highgui.CV_CAP_PROP_FRAME_WIDTH, width )
opencv.highgui.cvSetCaptureProperty( camera, opencv.highgui.CV_CAP_PROP_FRAME_HEIGHT, height )
opencv.highgui.cvSetCaptureProperty( camera, opencv.highgui.CV_CAP_PROP_SATURATION, 0.0 )

pygame.init()
window = pygame.display.set_mode( ( width, height ) )
pygame.display.set_caption( "FOUCAM PROCESSING DEVELOPMENT" )
screen = pygame.display.get_surface()
font = pygame.font.Font( None, 36 )
background = pygame.Surface( screen.get_size() )
background = background.convert()
background.fill( ( 250, 250, 250 ) )


while True:
    faces = []
    events = pygame.event.get()
    for event in events:
        if event.type == QUIT or event.type == KEYDOWN:
            sys.exit(0)

    img_PIL =  getimage.get_image_PIL( camera )
    background.blit( getimage.convert_PIL_to_pygame(img_PIL), (0, 0) )
    
    for face in faces:
        pygame.draw.rect( background, (0, 255, 0), ( face[0], face[1], face[2], face[3]), 2 ) 
    
    screen.blit( background, ( 0, 0 ) )
    screen.blit( font.render( 'FOUCAM', 1, ( 255, 255, 255 ) ), ( 10, 10 ) )
    pygame.display.flip()
