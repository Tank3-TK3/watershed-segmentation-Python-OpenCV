#############################################################################################
#                                          MODULOS
import numpy as np
import cv2 
from matplotlib import pyplot as plt
from collections import deque
#############################################################################################
#                                         FUNCIONES
def doThresh( i , umb , fondo , obj ):
    thresh1 = np.zeros( i.shape , dtype=np.uint8 )
    for a in range( 0 , i.shape[0]):
        for b in range( 0 , i.shape[1]):
            num = np.uint8( ( int( i[a][b][0] ) + int( i[a][b][1] ) + int( i[a][b][2] ) )/3 )
            if num >= umb:
                thresh1[a][b][0] = obj
                thresh1[a][b][1] = obj
                thresh1[a][b][2] = obj
            else:
                thresh1[a][b][0] = fondo
                thresh1[a][b][1] = fondo
                thresh1[a][b][2] = fondo
    return thresh1

def vecinos( im , i , j ):
    m = []
    for y in range( -1 , 2 ):
        for x in range( -1 , 2 ):
            try:
                m.append( im[i+x][j+y][0] )
            except:
                pass
    m.sort()
    return m

def filtroMediana( imgOrg , imgRuido ):
    imgMediana = np.zeros( imgOrg.shape , dtype=np.uint8 )
    mask = []
    for i in range( 0 , imgOrg.shape[0] ):
        for j in range( 0 , imgOrg.shape[1] ):
            mask = vecinos( imgRuido , i , j )
            imgMediana[i][j][0] = mask[np.uint8( len( mask ) / 2 )]
            imgMediana[i][j][1] = mask[np.uint8( len( mask ) / 2 )]
            imgMediana[i][j][2] = mask[np.uint8( len( mask ) / 2 )]
            mask = []
    return imgMediana

def dilatacion( im ):
    imgDilatada = np.zeros( im.shape , dtype=np.uint8 )
    for i in range( 0 , im.shape[0] ):
        for j in range( 0 , im.shape[1] ):
            if im[i][j][0] == 255:
                for y in range( -1 , 2 ):
                    for x in range( -1 , 2 ):
                        try:
                            imgDilatada[i+x][j+y][0] = np.uint8(255)
                            imgDilatada[i+x][j+y][1] = np.uint8(255)
                            imgDilatada[i+x][j+y][2] = np.uint8(255)
                        except:
                            pass
    return imgDilatada
#############################################################################################
#                                           MAIN
if __name__ == '__main__':
    name1 = './img/water_coins.jpg'
    name2 = './img/objetos6.jpg'

    #Leer la imagen
    gray  = cv2.imread( name2 )
    plt.subplot( 1 , 1 , 1 )
    plt.imshow( gray )
    plt.title('IMG Original')
    plt.axis( 'off' )
    plt.show()

    #Umbralizacion 
    thresh = doThresh( gray  , umb=128 , fondo=255 , obj=0 )
    plt.subplot( 2 , 3 , 1 )
    plt.imshow( thresh )
    plt.title('IMG thresh')
    plt.axis( 'off' )

    #Eliminacion de Ruido por Filtro de Mediana
    opening = filtroMediana( gray  , thresh )
    plt.subplot( 2 , 3 , 2 )
    plt.imshow( opening )
    plt.title('IMG sin ruido')
    plt.axis( 'off' )

    #Dilatacion de la imagen
    sure_bg = dilatacion( opening )
    plt.subplot( 2 , 3 , 3 )
    plt.imshow( sure_bg )
    plt.title('IMG sure_bg')
    plt.axis( 'off' )  

    #Distacia
    opening = cv2.cvtColor( opening , cv2.COLOR_BGR2GRAY )
    dist_transform = cv2.distanceTransform( opening , cv2.DIST_L2  , 3 )
    plt.subplot( 2 , 3 , 4 )
    plt.imshow( dist_transform , 'gray' )
    plt.title( 'IMG dist_transform' )
    plt.axis( 'off' )

    #Umbralizacion 2
    dist_transform = np.uint8( dist_transform )
    dist_transform2 = np.zeros( gray.shape , dtype=np.uint8 )
    for i in range( 0 , gray.shape[0]):
        for j in range( 0 , gray.shape[1]):
            dist_transform2[i][j][0] = dist_transform[i][j]
            dist_transform2[i][j][1] = dist_transform[i][j]
            dist_transform2[i][j][2] = dist_transform[i][j]
    sure_fg = doThresh( dist_transform2 , umb=3 , fondo=0 , obj=255 )
    plt.subplot( 2 , 3 , 5 )
    plt.imshow( sure_fg , 'gray' )
    plt.title('IMG sure_fg')
    plt.axis( 'off' )

    unknown = cv2.subtract( sure_bg , sure_fg )
    plt.subplot( 2 , 3 , 6 )
    plt.imshow( unknown , 'gray' )
    plt.title('IMG unknown')
    plt.axis( 'off' )

    plt.show()

    sure_fg = cv2.cvtColor( sure_fg , cv2.COLOR_BGR2GRAY )
    ret, markers = cv2.connectedComponents( sure_fg )
    plt.subplot( 1 , 3 , 1 )
    plt.imshow( markers )
    plt.title('IMG markers')
    plt.axis( 'off' )

    markers = markers+1
    unknown = cv2.cvtColor( unknown , cv2.COLOR_BGR2GRAY )
    markers[unknown==255] = 0
    markers = cv2.watershed( gray , markers )
    plt.subplot( 1 , 3 , 2 )
    plt.imshow( markers )
    plt.title('IMG markers')
    plt.axis( 'off' )

    gray[markers == -1] = [255,0,0]
    plt.subplot( 1 , 3 , 3 )
    plt.imshow( cv2.cvtColor( gray , cv2.COLOR_BGR2RGB )  )
    plt.title( 'Coins Img con markers' )
    plt.axis( 'off' )
    
    plt.show()