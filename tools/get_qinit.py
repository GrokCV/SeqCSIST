import scipy.io as sio
import numpy as np
import os
import re
from gen_annotation import read_bounding_boxes_from_xml

def extract_number(filename):
    # 使用正则表达式在文件名中查找数字
    match = re.search(r'\d+', filename)
    
    # 如果找到了数字，提取并返回该数字
    # 否则，返回 -1
    return int(match.group()) if match else -1

# 读取 xml 文件，将其转换为矩阵
def xml_2_matrix_single(xml_file):
  targets_GT, *_ = read_bounding_boxes_from_xml(xml_file)
  A = np.zeros((33, 33))
  for i in range(len(targets_GT)):
    x, y, lightness = targets_GT[i][0], targets_GT[i][1], targets_GT[i][2]
    A[int(round(3 * x + 1, 0)), int(round(3 * y + 1, 0))] = lightness
  return A

# 读取 xml_root 路径下的所有 xml 文件，将其转换为矩阵
def xml_2_matrix(xml_root):
  x = []
  for xml_file in sorted(os.listdir(xml_root), key=extract_number):
    print(xml_file)
    A = xml_2_matrix_single(os.path.join(xml_root, xml_file))     
    x.append(A.reshape(1, 1089))
  return x

# 矩阵初始化
def initialization(initial_matrix_root):
  Qinit_Name = initial_matrix_root
  # Computing Initialization Matrix:
  if os.path.exists(Qinit_Name):
      print("----------------Qinit 存在 --------------------------")
  else:
      # 读取矩阵 phi
      Phi_data_Name = '/opt/data/private/Simon/DeRefNet/data/phi_0.5.mat'
      Phi_data = sio.loadmat(Phi_data_Name)
      Phi_input = Phi_data['phi']

      # 读取读取 xml 文件并将训练数据转换为矩阵形式
      Training_labels = xml_2_matrix("/opt/data/private/Simon/DeRefNet/data/track_5000_20/train/annotation")

      # 计算初始化矩阵，根据最小二乘法的公式，Qinit = X * Y^T * (Y * Y^T)^(-1)
      X_data = np.squeeze(Training_labels).T
      print(X_data.shape)
      Y_data = np.dot(Phi_input, X_data)
      Y_YT = np.dot(Y_data, Y_data.transpose())
      X_YT = np.dot(X_data, Y_data.transpose())
      # 矩阵求逆
      Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
      # 删除临时变量，节省内存
      del X_data, Y_data, X_YT, Y_YT
      # 保存初始化矩阵
      sio.savemat(Qinit_Name, {'Qinit': Qinit})
      print("generate done")

file_path = '/opt/data/private/Simon/DeRefNet/data/track_5000_20/train/qinit.mat'
initialization(file_path)