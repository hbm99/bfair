from PIL import Image


def _get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
    min_height = min(im.height for im in im_list)
    im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height),resample=resample)
                      for im in im_list]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst

def _get_concat_v_multi_resize(im_list, resample=Image.BICUBIC):
    min_width = min(im.width for im in im_list)
    im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)),resample=resample)
                      for im in im_list]
    total_height = sum(im.height for im in im_list_resize)
    dst = Image.new('RGB', (min_width, total_height))
    pos_y = 0
    for im in im_list_resize:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    return dst

def _get_concat_tile_resize(im_list_2d, resample=Image.BICUBIC):
    im_list_v = [_get_concat_h_multi_resize(im_list_h, resample=resample) for im_list_h in im_list_2d]
    return _get_concat_v_multi_resize(im_list_v, resample=resample)

def concat_images(image_list):
        return _get_concat_tile_resize(image_list)

def merge_attribute_values(attribute, row_i, row_j):
    attribute_value_i = row_i[attribute]
    attribute_value_j = row_j[attribute]
    if isinstance(attribute_value_i, list) and attribute_value_j not in attribute_value_i:
        attribute_value_i.append(attribute_value_j)
    elif isinstance(attribute_value_i, str) and attribute_value_i != attribute_value_j:
        attribute_value_i = [attribute_value_i, attribute_value_j]
    return attribute_value_i