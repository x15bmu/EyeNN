import numpy as np

def crop_repeating_edge(image, rect):
    crop_x, crop_y, crop_w, crop_h = [int(round(c)) for c in rect]

    left_padding = max(0, -crop_x)
    top_padding = max(0, -crop_y)
    right_padding = max(crop_x + crop_w - image.shape[1], 0)
    bottom_padding = max(crop_y + crop_h - image.shape[0], 0)


    output = np.zeros((crop_h, crop_w, image.shape[2]))

    content_out_pixels_y = slice(top_padding, crop_h - bottom_padding)
    content_out_pixels_x = slice(left_padding, crop_w - right_padding)
    content_in_pixels_y = slice(crop_y + top_padding, crop_y + crop_h - bottom_padding)
    content_in_pixels_x = slice(crop_x + left_padding, crop_x + crop_w - right_padding)
    output[content_out_pixels_y, content_out_pixels_x, :] =\
            image[content_in_pixels_y, content_in_pixels_x]

    # Check for errors that occurred
    if content_out_pixels_x.stop - content_out_pixels_x.start <= 0:
        raise Exception('No out pixels in x direction')
    if content_out_pixels_y.stop - content_out_pixels_y.start <= 0:
        raise Exception('No out pixels in y direction')

    # Pad directly above and below image
    if top_padding > 0:
        output[:top_padding, content_out_pixels_x, :] =\
                np.tile(output[np.newaxis, content_out_pixels_y.start, content_out_pixels_x, :],\
                        (top_padding, 1, 1))
    if bottom_padding > 0:
        output[-bottom_padding:, content_out_pixels_x, :] =\
                np.tile(output[np.newaxis, content_out_pixels_y.stop - 1, content_out_pixels_x, :],\
                        (bottom_padding, 1, 1))

    # Pad to the left and right
    if left_padding > 0:
        output[:, :left_padding, :] =\
                np.tile(output[:, content_out_pixels_x.start, np.newaxis, :],\
                        (1, left_padding, 1))
    if right_padding > 0:
        output[:, -right_padding:, :] =\
                np.tile(output[:, content_out_pixels_x.stop - 1, np.newaxis, :],\
                        (1, right_padding, 1))

    return output

