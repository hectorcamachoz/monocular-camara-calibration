"""get-mesurments.py
    This python script is used to get mesurments from a live video stream.

    Author: Héctor Camacho Zamora
    Organisation: Universidad de Monterrey
    Contact: hector.camacho.zamora@udem.edu
    First created: Monday 1 Abril 2024
    Last updated: 
    
    Example of usage:
    python3 get-mesurments.py --cam_index 0 --Z 56 --cal_file calibration-parameters/calibration_data.json

"""
# Import standard libraries
import numpy as np
import cv2
import glob
import os
import argparse
import sys
import textwrap
import json
import platform
import math
from numpy.typing import NDArray
from typing import List, Tuple

# Import Local Libraries
import correct_image_distortion as correct
import monocular_camera_calibration_helpers as helpers


points = []
coords = []
lines = []
left_clk_block = False 
wh = True
lines_calculated = False
def parse_data_from_live_camera_mesurments()->argparse.ArgumentParser:
    """
    Parse command-line arguments for camera calibration.

    Returns:
        argparse.ArgumentParser: The argument parser object configured for camera calibration.
    """

    # Parse user's argument
    parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter,
                                    description=textwrap.dedent('''\
            This Python script performs .....

            '''))
    parser.add_argument('-i',
                        '--cam_index', 
                        type=int, 
                        required=True,
                        default='0',
                        help="Camara you are using ")
    parser.add_argument('-z',
                        '--Z',
                        type=str,
                        required=True,
                        help="Distance from camera to the object")
    parser.add_argument('-c',
                        '--cal_file',
                        type=str,
                        required=True,
                        help='Path to JSON file containing the camera calibration parameters')
    args = parser.parse_args()

    return args


def initialise_camera(args:argparse)->cv2.VideoCapture:
    
    """
    This function initialises the camera to capture the video
    
    Args:
        args (argparse): The command line arguments

    Returns:
        cv2.VideoCapture: The video capture object
    """
    # Create a video capture object
    cap = cv2.VideoCapture(args.cam_index)
    
    return cap

def load_calibration_parameters_from_json_file(
        args:argparse.ArgumentParser
        )->None:
    """
    Load camera calibration parameters from a JSON file.

    Args:
        args: Parsed command-line arguments.

    Returns:
        camera_matrix: Camera matrix.
        distortion_coefficients: Distortion coefficients.

    This function may raise a warning if the JSON file 
    does not exist. In such a case, the program finishes.
    """

    # Check if JSON file exists
    json_filename = args.cal_file
    check_file = os.path.isfile(json_filename)

    # If JSON file exists, load the calibration parameters
    if check_file:
        f = open(json_filename)
        json_data = json.load(f)
        f.close()
        
        camera_matrix = np.array(json_data['camera_matrix'])
        distortion_coefficients = np.array(json_data['distortion_coefficients'])
        return camera_matrix, distortion_coefficients
    
    # Otherwise, the program finishes
    else:
        print(f"The file {json_filename} does not exist!")
        sys.exit(-1)

def undistort_images(
        frame:cv2, 
        mtx:NDArray, 
        dist:NDArray, 
        )->cv2.VideoCapture:
    """
    Undistort a list of distorted images using camera calibration parameters and save 
    the undistorted images.

    Args:
        list_of_undistorted_images: List of paths to distorted images.
        mtx: Camera matrix.
        dist: Distortion coefficients.
        path_to_saving_undistorted_images: Path to save undistorted images.

    Returns:
        None: The function does not return any value.
    """
   
    img = frame
    # Get size
    h,  w = img.shape[:2]

    # Get optimal new camera
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # Undistort image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Crop image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst        

def draw(im_copy:cv2, points):
  
        for point in points:
            
          
            cv2.circle(im_copy,(point[0],point[1]), 5, (255,0,0), -1)
            cv2.waitKey(10)

        for i in range(len(points) - 1):
            cv2.line(im_copy, points[i], points[i+1], (0, 0, 255), 3)
            if left_clk_block == True:
                p = len(points)-1
                cv2.line(im_copy, points[p], points[0], (0, 0, 255), 3)

def compute_line_segments(im_copy:cv2, mtx:NDArray, dist:NDArray,points):
    global left_clk_block
    global wh
    
    global lines_calculated 
    while wh == True:
        if left_clk_block == True :
            
            for i in range(len(points)):
                if len(points) == len(coords):
                    break
                coor_X = (np.float64(points[i][0]) - mtx[0,2]) * np.float64(args.Z) / mtx[0,0]
                coor_Y = (np.float64(points[i][1]) - mtx[1,2]) * np.float64(args.Z) / mtx[1,1]
                coords.append( [coor_X,coor_Y,args.Z] )
            
            if len(coords) > 1:
                
                for j in range(len(coords)):
                    if len(coords) == 2:
                        ln = math.sqrt((abs(coords[0][0]) - abs(coords[1][0]))**2 
                                        + (abs(coords[0][1]) - abs(coords[1][1]))**2)
                        lines.append(ln)
                        lines_calculated = True  
                        print('Linea ', j, ': ', lines[j])
                        break
                    if j == len(coords)-1:
                        ln = math.sqrt((abs(coords[0][0]) - abs(coords[j][0]))**2 
                                        + (abs(coords[0][1]) - abs(coords[j][1]))**2)
                        lines.append(ln)
                        lines_calculated = True  
                        print('Linea ', j, ': ', lines[j])
                        break

                    if len(lines)<=len(coords)-1 and lines_calculated == False:
                        ln = math.sqrt((abs(coords[j+1][0]) - abs(coords[j][0]))**2 
                                        + (abs(coords[j+1][1]) - abs(coords[j][1]))**2)
                        lines.append(ln)
                    print('Linea ', j, ': ', lines[j])
                lines_sorted = sorted(lines, reverse=True)
        
            # Imprimir las distancias de las líneas ordenadas
            print("\nLongitudes de líneas ordenadas de mayor a menor:\n")
            
            for line in lines_sorted:
                index = lines.index(line)
                print(f"Linea {index}: {line}") 
                
        wh = False

def compute_perimeter(lines):
    global lines_calculated
    while lines_calculated == True:
        perimeter = 0
        for i in range(len(lines)):
            perimeter = perimeter + lines[i]
        if len(lines) != 1:
            print("\nPerimetro de la figura: ",perimeter)
            print("\n- Presione en pantalla la tecla 'ctrl' para borrar los puntos y tomar otra figura.",
                "\n- Presione en pantalla la tecla 'q' para salir del programa.\n")
        elif len(lines) == 1:
            print("Longitud de la linea: ",lines[0])
        lines_calculated = False

    
def Mouse_events(event, x, y, flags, param):
    global left_clk_block
    global wh

    
    i = len(points)
    if event != cv2.EVENT_RBUTTONDOWN and left_clk_block == False:
        if event == cv2.EVENT_LBUTTONDOWN:
            print('Punto ',i, ' colocado')
            points.append([x,y])

          
            
    if event == cv2.EVENT_MBUTTONDOWN:
        if left_clk_block == False:
            if len(points) > 1:
                print('\nCalculando longitud de lineas...\n')
                left_clk_block = True  
                wh = True
               
            else: 
                print('No hay suficientes puntos')
      

    if flags & cv2.EVENT_FLAG_CTRLKEY:
        print('Borrando puntos...\n\n')
        points.clear()
        coords.clear()
        lines.clear()
        left_clk_block = False
        wh = False
        lines_calculated = False
        run_pipeline(args)
    
def run_pipeline(args:argparse.ArgumentParser)->None:

    cap = initialise_camera(args)
    camera_matrix, distortion_coefficients = load_calibration_parameters_from_json_file(args)

    cv2.namedWindow('Undistorted Live Camara',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Undistorted Live Camara', Mouse_events)
    #cv2.setMouseCallback('Undistorted Live Camara', KeyboardEvent)
    size = 640,360
    print("\n-- BIENVENIDO --")
    print("\n- Presione en pantalla la tecla 'q' para salir del programa.",
        "\n- Presione en pantalla la rueda del raton para terminar de seleccionar los puntos.\n")
    while cap.isOpened():
        ret, frame = cap.read()
        
        dst = undistort_images(frame, 
                     camera_matrix, 
                     distortion_coefficients)
        dst = cv2.resize(dst, size)
        im_copy = dst.copy()
        
        compute_line_segments(im_copy,camera_matrix, 
                              distortion_coefficients, 
                              points)
        draw(im_copy, points)
        compute_perimeter(lines)
        cv2.imshow('Undistorted Live Camara',im_copy)
        

        key = cv2.waitKey(1) & 0xFF
        # Press Q on keyboard to  exit 
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_data_from_live_camera_mesurments()
    run_pipeline(args)
