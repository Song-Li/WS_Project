import csv
import numpy as np
import os
import random

def read_input(file_name):
  # file_name = "aidata.log"
  X = []
  Y = []
  with open( file_name , "r") as file:
    for line in file:
      line = line.replace("[", "")
      line = line.replace("]", ",")
      line = line.replace("True", "1")
      line = line.replace("False", "0")
      lst = line.split(",")
      X.append(lst[:-1])
      Y.append(lst[-1].strip())
    # print(lst[-1])
  X = np.array(X).astype(int)
  Y = np.array(Y).astype(int)
  return X, Y
def random_split(X, Y, split_ratio):
  split_tr = int(np.ceil(len(X) * split_ratio))
  split_t = len(X) - split_tr
  train_data = []
  val_data = []
  train_labels = []
  val_labels = []

  for i in range(len(Y)):
    choice = random.random()
    if choice >= split_ratio:
      val_data.append(X[i])
      val_labels.append(Y[i])
    else:
      train_data.append(X[i])
      train_labels.append(Y[i])

  train_data = np.array(train_data)
  val_data = np.array(val_data)
  train_labels = np.array(train_labels)
  val_labels = np.array(val_labels)
  return train_data, train_labels, val_data, val_labels

def write_data(train_data, train_labels, val_data, val_labels, td_path, tl_path, vd_path, vl_path):
  with open(td_path, mode='a') as file:
    writer = csv.writer(file)
    writer.writerows(train_data)
  with open(tl_path, mode='a') as file:
    writer = csv.writer(file)
    writer.writerow(train_labels)

  with open(vd_path, mode='a') as file:
    writer = csv.writer(file)
    writer.writerows(val_data)
  with open(vl_path, mode='a') as file:
    writer = csv.writer(file)
    writer.writerow(val_labels)

def read_data(data_file, labels_file):
  X = []
  Y = []
  with open( data_file , "r") as file:
    for line in file:
      lst = line.split(",")
      lst[-1] = lst[-1].strip()
      X.append(lst)
      # print(lst.shape)
    # print(lst[-1])
  with open( labels_file , "r") as file:
    for line in file:
      lst = line.split(",")
      lst[-1] = lst[-1].strip()
      # print(lst)
      Y = (lst)
  X = np.array(X).astype(int)
  Y = np.array(Y).astype(int)
  return X, Y