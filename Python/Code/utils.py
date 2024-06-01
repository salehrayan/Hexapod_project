import xml.etree.ElementTree as ET



def parse_urdf_for_colors(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    colors = []
    for link in root.findall('link'):
        visual = link.find('visual')
        if visual is not None:
            material = visual.find('material')
            if material is not None:
                color = material.find('color')
                if color is not None:
                    rgba = color.attrib['rgba'].split()
                    colors.append((float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])))
                else:
                    colors.append((0.5, 0.5, 0.5, 1.0))  # Default
            else:
                colors.append((0.5, 0.5, 0.5, 1.0))  # Default
        else:
            colors.append((0.5, 0.5, 0.5, 1.0))  # Default
    return colors







