import os
import xml.etree.ElementTree as ET

# Define directories
xml_dir = 'C:/Users/WB GAMING/Desktop/pothole/archive (7)/annotations/'  # Directory containing XML files
yolo_dir = 'C:/Users/WB GAMING/Desktop/pothole/archive (7)/outputs'  # Directory to save YOLO .txt files

# Ensure YOLO output directory exists
os.makedirs(yolo_dir, exist_ok=True)

# Define a class name list (customize as needed)
class_names = ["pothole"]  # Replace with your actual class names

# Function to convert XML to YOLO format
def convert_xml_to_yolo(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    # YOLO formatted annotations to write in the txt file
    yolo_annotations = []

    # Loop through each object in XML
    for obj in root.iter('object'):
        # Get class name and convert to class ID
        class_name = obj.find('name').text
        if class_name not in class_names:
            continue
        class_id = class_names.index(class_name)

        # Get bounding box coordinates
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # Convert to YOLO format (normalize values)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Format: class_id x_center y_center width height
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return yolo_annotations

# Loop through all XML files in the directory
for xml_filename in os.listdir(xml_dir):
    if xml_filename.endswith('.xml'):
        # Read XML file and convert annotations
        xml_path = os.path.join(xml_dir, xml_filename)
        yolo_annotations = convert_xml_to_yolo(xml_path)
        
        # Write YOLO annotations to corresponding .txt file
        yolo_filename = os.path.splitext(xml_filename)[0] + '.txt'
        yolo_path = os.path.join(yolo_dir, yolo_filename)
        with open(yolo_path, 'w') as yolo_file:
            yolo_file.write('\n'.join(yolo_annotations))

print("Conversion complete!")
