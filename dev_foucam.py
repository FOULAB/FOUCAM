#!/usr/bin/env python

import os
import sys

import pygame
from pygame.locals import *

import opencv 
from opencv import highgui

try:
    import psyco
    psyco.full()
except ImportError:
    pass

pygame.init()


class CameraInterface( object ):
 
    def __init__( self, resolution ):
        self._camHW = opencv.highgui.cvCreateCameraCapture(0)
        highgui.cvSetCaptureProperty( self._camHW, opencv.highgui.CV_CAP_PROP_FRAME_WIDTH, resolution[ 'WIDTH' ] )
        highgui.cvSetCaptureProperty( self._camHW, opencv.highgui.CV_CAP_PROP_FRAME_HEIGHT, resolution[ 'HEIGHT' ] )
        highgui.cvSetCaptureProperty( self._camHW, opencv.highgui.CV_CAP_PROP_SATURATION, 0.0 )

        self._iplFrame = None

    def __del__( self ):
        highgui.cvSetCaptureProperty( self._camHW, highgui.CV_CAP_PROP_SATURATION, 1.0 )

    def pollCamera( self ):
        self._iplFrame = highgui.cvQueryFrame( self._cameraHW )
        opencv.cv.cvMirror( self._iplFrame, None, 1 ) 

    def getFrameAsIpl( self ):
        return self._iplFrame
    
    def getFrameAsPIL( self ):
        return opencv.adaptors.Ipl2PIL( self._iplFrame )

    def getFrameAsPygame( self ):
        image_PIL = self.getFrameAsPIL()
        return pygame.image.frombuffer( image_PIL.tostring(), image_PIL.size, image_PIL.mode )
    

class FOUCAM( object ):
    
    def __init__( self, **argd ):
        self._resolution = { 'WIDTH': 640,
                             'HEIGHT': 480 }

        self.__dict__.update( **argd )
        super( FOUCAM, self ).__init__( **argd ) 

        self._camera = CameraInterface( self._resolution )

        pygame.display.set_mode( ( self._resolution['WIDTH'], self._resolution['HEIGHT'] ) )
        pygame.display.set_caption( "FOUCAM PIPELINE PROTOTYPE" )
        self._font = pygame.font.Font( None, 36 )
        
        self._screen = pygame.display.get_surface()

        self._background = pygame.Surface( self._screen.get_size() )
        self._background = self._background.convert()
        self._background.fill( ( 250, 250, 250 ) )

        self._trainedHaar = 'haarcascade_frontalface.xml'
       
        self._faces = []

    def __del__( self ):
        pass

    def pollCamera( self ):
        self._camera.pollCamera()
        #opencv.cvSmooth( img, img, opencv.CV_GAUSSIAN, 9, 9 )
        self._background.blit( self._camera.getFrameAsPygame(), (0, 0) )

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
