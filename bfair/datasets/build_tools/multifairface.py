

def concat_images(image_list):
        pass

def merge_attribute_values(attribute, row_i, row_j):
    attribute_value_i = row_i[attribute]
    attribute_value_j = row_j[attribute]
    if isinstance(attribute_value_i, list) and attribute_value_j not in attribute_value_i:
        attribute_value_i.append(attribute_value_j)
    elif isinstance(attribute_value_i, str) and attribute_value_i != attribute_value_j:
        attribute_value_i = [attribute_value_i, attribute_value_j]
    return row_i[attribute]