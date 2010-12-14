#!/usr/bin/python

# This script takes in a directory of output from the
# manual placement program and generates a training set
# for the neural network

import sys, os
import math

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __str__(self):
    return "(" + str(self.x) + ", " + str(self.y) + ")"

def parse_file(filename):
  f = open(filename, "r")
  parse=dict() 
  for line in f:
    #Really need to get better with regex in python
    [tag, coord] = line.strip().split(":")
    [x, y] = coord.split()
    x = int(x.split("=")[1])
    y = int(y.split("=")[1])
    p = Point(x, y)
    parse[tag] = p
  return parse

def build_file_list(arg, current, files):
  flist=list()
  for x in files:
    if os.path.isfile(current + "/" +  x):
      flist.append(current + "/" + x)
  arg.append(os.path.basename(current), flist)
 
class TrainingFileList:
  emotion_data = dict()

  def append(self, emotion, files):
    if(files != list()):
      self.emotion_data[emotion] = files

def dist(p1, p2):
  return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

def calc_distance_feature(dataset):
  subject_list = dict()
  for subject in dataset:
    fv = dataset[subject]
    d_mouth_w = dist(fv["mouth_left"], fv["mouth_right"])
    d_mouth_h = dist(fv["mouth_top"], fv["mouth_bottom"])
   
    d_left_eye = dist(fv["lefteye_top"], fv["lefteye_bottom"])
    d_right_eye = dist(fv["righteye_top"], fv["righteye_bottom"])
 
    d_left_brow_left = dist(fv["leftbrow_left"], fv["nose"])
    d_left_brow_middle = dist(fv["leftbrow_middle"], fv["nose"])
    d_left_brow_right = dist(fv["leftbrow_right"], fv["nose"])
 
    d_right_brow_left = dist(fv["rightbrow_left"], fv["nose"])
    d_right_brow_middle = dist(fv["rightbrow_middle"], fv["nose"])
    d_right_brow_right = dist(fv["rightbrow_right"], fv["nose"])

    features = dict()
    features["d_mouth_w"]        = d_mouth_w
    features["d_mouth_h"]        = d_mouth_h
    features["d_left_eye"]       = d_left_eye
    features["d_right_eye"]      = d_right_eye
    features["d_left_brow_left"] = d_left_brow_left
    features["d_left_brow_middle"] = d_left_brow_middle
    features["d_left_brow_right"] = d_left_brow_right
    features["d_right_brow_left"] = d_right_brow_left
    features["d_right_brow_middle"] = d_right_brow_middle
    features["d_right_brow_right"] = d_right_brow_right
    subject_list[subject] = features

  return subject_list

def  write_training_data(filename, *args):
  outputs  = len(args)
  samples  = len(args[0])
  inputs   = len(args[0][args[0].keys()[0]]) #Gotta be a better way!!

  samples = outputs*samples

  f = open(filename, "w")
  f.write(str(samples)+" "+str(inputs)+" "+str(outputs)+"\n")

  output_num = 0

  for emotion in args:
    for subject in emotion:
      dump=""
      for x in emotion[subject]:
        dump += (str(emotion[subject][x])+" ")
      dump += "\n"
      f.write(dump)

      output = list()
      for x in xrange(outputs):
        output.append(0)
      output[output_num] = 1
  
      dump=""
      for x in output:
        dump += str(x) + " "
      dump += "\n"
      f.write(dump)
 
    output_num += 1


  f.close()

def calc_normalized_features(emotion, neutral):
  subjects = dict()
  for subject in neutral:
    normalized = dict()
    for x in neutral[subject]:
      normalized[x] = emotion[subject][x]/neutral[subject][x]
 
    subjects[subject] = normalized
  return subjects

def main():
  files = TrainingFileList()
  os.path.walk(sys.argv[1], build_file_list, files) 
 
  for x in files.emotion_data:
    print "Found " + str(len(files.emotion_data[x])) + " training files for \"" + x + "\"" 
 
  neutral = dict()
  print "Parsing Neutral Emotion Data Set..."
  for x in files.emotion_data["Neutral"]:
    neutral[os.path.basename(x)] = parse_file(x)

  happiness = dict()
  print "Parsing Happy Emotion Data Set..."
  for x in files.emotion_data["Happiness"]:
    happiness[os.path.basename(x)] = parse_file(x)

  surprise = dict()
  print "Parsing Surprise Emotion Data Set..."
  for x in files.emotion_data["Surprise"]:
    surprise[os.path.basename(x)] = parse_file(x)

  sadness = dict()
  print "Parsing Sadness Emotion Data Set..."
  for x in files.emotion_data["Sadness"]:
    sadness[os.path.basename(x)] = parse_file(x)

  print "Calculating distance features.."
  happy_dist    = calc_distance_feature(happiness)
  sad_dist      = calc_distance_feature(sadness)
  surprise_dist = calc_distance_feature(surprise)
  neutral_dist  = calc_distance_feature(neutral)

  print "Calculating normalized features.."
  happy_normal    = calc_normalized_features(happy_dist, neutral_dist)
  sad_normal      = calc_normalized_features(sad_dist, neutral_dist)
  surprise_normal = calc_normalized_features(surprise_dist, neutral_dist)
  neutral_normal  = calc_normalized_features(neutral_dist, neutral_dist) 

  print "Writing training data..."
  write_training_data("training.dat", happy_normal, sad_normal, surprise_normal, neutral_normal)

if __name__ == "__main__":
  main()

