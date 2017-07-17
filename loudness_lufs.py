#!/usr/bin/env python

import argparse
import os
import subprocess
import re

"""
This script serves to return a dict of parmeters concerning loudness measurement, including LUFS. 
To run the script, type "python loudness_lufs.py -i /path/to/file.wav -s 0 -t -16"
The path is the only thing you should fill in.
"""

def readFile(filep):
    """Funciton that reads file from path as a byte object

    Args:
        filep: the absolute path of the file to be opened

    Returns:
        Byte: The opened file as a byte object.   
    """
    f = open(filep, 'rb')
    output = f
    f.close()
    return output

# return a collection of loudness parameters
def r128Stats(filePath, stream=0):
    """Function that takes a path to an audio file, returns a dict with the loudness
    stats computed by the ffmpeg ebur128 filter 
    
    Args:
        filePath (str): the absolute path of the file
        stream (int): the stream value

    Returns:
        dict: It has 7 key-value pairs.

    """
    ffargs = ["ffmpeg",
              '-nostats',
              '-i',
              filePath,
              '-filter_complex',
              '[a:%s]ebur128' % stream,
              '-f',
              'null',
              '-']
    proc = subprocess.Popen(ffargs, stderr=subprocess.PIPE)
    stats = proc.communicate()[1]
    summaryIndex = stats.rfind(b'Summary:')
    summaryList = stats[summaryIndex:].decode('utf8').split()


    ILufs = float(summaryList[summaryList.index('I:') + 1])
    IThresh = float(summaryList[summaryList.index('I:') + 4])
    LRA = float(summaryList[summaryList.index('LRA:') + 1])
    LRAThresh = float(summaryList[summaryList.index('LRA:') + 4])
    LRALow = float(summaryList[summaryList.index('low:') + 1])
    LRAHigh = float(summaryList[summaryList.index('high:') + 1])
    statsDict = {'I': ILufs, 'I Threshold': IThresh, 'LRA': LRA,
                 'LRA Threshold': LRAThresh, 'LRA Low': LRALow,
                 'LRA High': LRAHigh}
    return statsDict

# calculate the linear gain in the dictionary from the function above
def linearGain(iLUFS, goalLUFS=-16):
    """Function that takes a floating point value for iLUFS, returns the necessary
    multiplier for audio gain to get to the goalLUFS value
    
    Args: 
        iLUFS (float): The value of LUFS
        goalLUFS (int): The goal LUFS used for calculating difference of real LUFS.

    Returns:
        float: the value of linear gain

    """

    gainLog = -(iLUFS - goalLUFS)
    return 10 ** (gainLog / 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loudness Analyser.')
    parser.add_argument("-i", "--input", help="The video or audio filename ie /path/to/file/filename.mp4", required=True)
    parser.add_argument("-s", "--stream", help="The audio stream index", required=False, default=0)
    parser.add_argument("-t", "--target", help="The audio stream index", required=False, default=-16)
    args = parser.parse_args()

    if os.path.isfile(args.input):
        
        # print the LUFS value
        print("print args.target: ", args.target)        
        statsDict = r128Stats(args.input, args.stream)
        statsDict["gain"] = linearGain(statsDict["I"], int(args.target));
        print(statsDict['I'])
        print(statsDict)
        

    else:
        print("%s not found" % args.input)



