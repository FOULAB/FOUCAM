#!/usr/bin/env python

from ctypes import *

import os
import sys

import pygame
from pygame.locals import *

import opencv 
from opencv import highgui

import time
from multiprocessing import Process, Pool, Queue, Pipe

try:
    import psyco
    psyco.full()
except ImportError:
    pass

pygame.init()

# some useful functions
# convert the hue value to the corresponding rgb value
def hsv2rgb ( hue ):
    sector_data = [ [0, 2, 1],
                    [1, 2, 0],
                    [1, 0, 2],
                    [2, 0, 1],
                    [2, 1, 0],
                    [0, 1, 2] ]
    hue = hue * ( 0.1 / 3 )
    sector = int( hue )
    p = int( round ( 255 * ( hue - sector ) ) )
    if sector & 1:
        p ^= 255

    rgb = {}
    rgb [ sector_data [ sector ] [ 0 ] ] = 255
    rgb [ sector_data [ sector ] [ 1 ] ] = 0
    rgb [ sector_data [ sector ] [ 2 ] ] = p

    return opencv.cv.Scalar (rgb [2], rgb [1], rgb [0], 0)



class CameraInterface( object ):
 
    def __init__( self, resolution ):
        self._camHW = opencv.highgui.cvCreateCameraCapture(0)
        highgui.cvSetCaptureProperty( self._camHW, opencv.highgui.CV_CAP_PROP_FRAME_WIDTH, resolution[ 'WIDTH' ] )
        highgui.cvSetCaptureProperty( self._camHW, opencv.highgui.CV_CAP_PROP_FRAME_HEIGHT, resolution[ 'HEIGHT' ] )
        #highgui.cvSetCaptureProperty( self._camHW, opencv.highgui.CV_CAP_PROP_SATURATION, 0.0 )

        self._iplFrame = None

    def __del__( self ):
        #highgui.cvSetCaptureProperty( self._camHW, highgui.CV_CAP_PROP_SATURATION, 1.0 )
        pass

    def nextFrame( self ):
        self._iplFrame = highgui.cvQueryFrame( self._camHW )
        opencv.cv.cvMirror( self._iplFrame, None, 1 ) 

    def getFrameAsIpl( self ):
        return self._iplFrame
    
    def getFrameAsPIL( self ):
        return opencv.adaptors.Ipl2PIL( self._iplFrame )

    def getFrameAsPygame( self ):
        framePIL = self.getFrameAsPIL()
        return pygame.image.frombuffer( framePIL.tostring(), framePIL.size, framePIL.mode )
    

class FOUCAM( object ):
    
    def __init__( self, **argd ):
        self._resolution = {}
        self._resolution[ 'WIDTH' ]  = 640
        self._resolution[ 'HEIGHT' ] = 480 

        self._camera = CameraInterface( self._resolution )

        pygame.display.set_mode( ( self._resolution['WIDTH'], self._resolution['HEIGHT'] ) )
        pygame.display.set_caption( "FOUCAM PIPELINE PROTOTYPE" )
        self._font = pygame.font.Font( None, 36 )
       
        self._screen = pygame.display.get_surface()

        self._background = pygame.Surface( self._screen.get_size() )
        self._background = self._background.convert()
        self._background.fill( ( 250, 250, 250 ) )

        self._trainedHaar = 'haarcascade_frontalface.xml'
       
        self._currentFrame = None

        self._blobs = []
        self._faces = []

    def __del__( self ):
        pass

    def pollCamera( self ):
        self._camera.nextFrame()
        #self._currentFrame = self._camera.getFrameAsIpl()
        #opencv.cvSmooth( img, img, opencv.CV_GAUSSIAN, 9, 9 )
        self._background.blit( self._camera.getFrameAsPygame(), (0, 0) )

    def detectBlobs( self ):
        self._blobs = []

    def detectFaces( self ):
        self._faces = []
        frame = self._camera.getFrameAsIpl()
        storage = opencv.cvCreateMemStorage( 0 )
        opencv.cvClearMemStorage( storage )
        cascade = opencv.cvLoadHaarClassifierCascade( self._trainedHaar, opencv.cvSize( 1, 1 ) )

        mugsht = opencv.cvHaarDetectObjects( frame,
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
                self._faces.append( face )


    def RUN( self ):
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == QUIT or ( event.type == KEYDOWN and event.key == K_ESCAPE ):
                    sys.exit(0)

            self.pollCamera()
            self.detectFaces()
            for face in self._faces:
                pygame.draw.rect( self._background, ( 0, 255, 0 ), ( face[0], face[1], face[2], face[3] ), 2 )
            
            self._screen.blit( self._background, ( 0, 0 ) )
            self._screen.blit( self._font.render( 'FOUCAM', 1, ( 255, 255, 255 ) ), ( 10, 10 ) )
            pygame.display.flip()


if __name__ == '__main__':
    FOUCAM().RUN()
