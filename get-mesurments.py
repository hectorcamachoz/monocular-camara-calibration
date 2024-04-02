"""get-mesurments.py
    This python script is used to get mesurments from a live video stream.

    Author: Héctor Camacho Zamora
    Organisation: Universidad de Monterrey
    Contact: hector.camacho.zamora@udem.edu
    First created: Monday 1 Abril 2024
    Last updated: 
    
    Example of usage:
    python get-measurements.py --cam_index 0 --Z 56 --cal_file calibration_data.json
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
from numpy.typing import NDArray
from typing import List, Tuple

# Import Local Libraries
import correct_image_distortion as correct
import monocular_camera_calibration_helpers as helpers

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


def run_pipeline(args:argparse.ArgumentParser)->None:
    
    cap = initialise_camera(args)
    camera_matrix, distortion_coefficients = load_calibration_parameters_from_json_file(args)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('Distorted Live Camara',frame)

        dst = undistort_images(frame, 
                     camera_matrix, 
                     distortion_coefficients)
        cv2.imshow('Undistorted Live Camara',dst)
        

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    args = parse_data_from_live_camera_mesurments()
    run_pipeline(args)
