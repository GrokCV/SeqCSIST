import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


# 存储图片信息
def save_image_infos(image_infos, image_id, w, h, sigma=0.5, depth="1",
                     xml_folder_location="data/CSO_11_100_10/Annotations"):
  # 创建一个 XML 根元素
  root = ET.Element("annotation")
  # 文件名
  image_name = "COS" + f"{image_id}"

  # 添加图片名称大小灰度信息以及 sigma
  file_name = ET.SubElement(root, "filename")
  file_name.text = image_name  # 图片名称
  size = ET.SubElement(root, "size")
  width = ET.SubElement(size, "width")
  width.text = f"{w}"  # 图片宽度
  height = ET.SubElement(size, "height")
  height.text = f"{h}"  # 图片高度
  height = ET.SubElement(size, "depth")
  height.text = depth  # 1 表示灰度图，3 表示 RGB 图
  argument = ET.SubElement(root, "sigma")
  argument.text = f"{sigma}"  # 高斯参数

  for image_info in image_infos:
    # 添加目标信息
    object_info = ET.SubElement(root, "object")
    name = ET.SubElement(object_info, "name")
    name.text = "Target"  # 目标名称
    coordinate = ET.SubElement(object_info, "coordinate")
    xc = ET.SubElement(coordinate, "xc")
    xc.text = image_info["xc"]
    yc = ET.SubElement(coordinate, "yc")
    yc.text = image_info["yc"]
    brightness = ET.SubElement(coordinate, "brightness")
    brightness.text = image_info["brightness"]

  # 保存 xml 文件
  xml_filename = os.path.join(xml_folder_location, image_name + f".xml")
  xml_str = ET.tostring(root, encoding="utf-8")
  dom = minidom.parseString(xml_str)
  pretty_xml_str = dom.toprettyxml()

  with open(xml_filename, "w", encoding="utf-8") as xml_file:
    xml_file.write(pretty_xml_str)


"""解析 XML 文件，获取单张图片的所有目标信息"""
def read_bounding_boxes_from_xml(xml_file_path):
  tree = ET.parse(xml_file_path)
  root = tree.getroot()
  targets_GT = []
  sigma = float(root.find('sigma').text)
  width = float(root.find('size').find('width').text)
  height = float(root.find('size').find('height').text)
  # 遍历所有 <object> 标签
  for obj in root.findall('object'):
    target_info = obj.find('coordinate')
    if target_info is not None:
      xc = float(target_info.find('xc').text)
      yc = float(target_info.find('yc').text)
      brightness = float(target_info.find('brightness').text)
      targets_GT.append([xc, yc, brightness])

  return targets_GT, sigma, width, height