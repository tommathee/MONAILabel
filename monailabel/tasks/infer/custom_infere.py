import numpy as np
import tensorflow as tf
import cv2


def get_fragment(image, x, y, patch_size, fill_value=255):
    if len(image.shape) == 3:
        height, width, channels = image.shape
        canvas = np.full((patch_size, patch_size, channels),
                         fill_value=fill_value)
    else:
        height, width = image.shape
        canvas = np.full((patch_size, patch_size), fill_value=fill_value)

    if x < 0 and y < 0 and patch_size > width and patch_size > height:  # DONE
        canvas[abs(y):abs(y) + height, abs(x): abs(x) +
               width] = image[0:height, 0:width]
    elif x < 0 and y < 0 and patch_size > width:  # DONE
        canvas[abs(y):patch_size, abs(x):abs(x) +
               width] = image[0:patch_size, 0:width]
    elif x < 0 and y < 0 and patch_size > height:
        canvas[abs(y):abs(y) + height, abs(x):patch_size] = image[0:height, 0:patch_size]
    elif x < 0 and y < 0:
        canvas[abs(y):patch_size, abs(
            x):patch_size] = image[0: patch_size, 0: patch_size]
    elif x < 0 and patch_size > width and patch_size > height:
        canvas[0:height, abs(x): abs(x) +
               width] = image[0:height, 0:width]
    elif x < 0 and patch_size > width:
        canvas[:, abs(x):abs(x) + width] = image[0:patch_size, 0:width]
    elif y < 0 and patch_size > height and patch_size > width:
        canvas[abs(y):abs(y) + height, 0:width] = image[0:height, x:]
    elif y < 0 and patch_size > width:
        canvas[abs(y):patch_size, 0:width] = image[0:patch_size, x:]
    elif y < 0 and patch_size > height:
        canvas[abs(y):abs(y) + height,
               :] = image[0:height, :patch_size]
    elif y < 0:
        canvas[abs(y):patch_size, :] = image[0:patch_size, 0:patch_size]
    elif patch_size > width and patch_size > height:
        canvas[0:height, 0:width] = image[0:height, 0:width]  # DONE
    elif x < 0 and patch_size > height:
        canvas[0:height, abs(x):patch_size] = image[0:height, 0:patch_size]
    elif patch_size > height:
        canvas[0: height, :] = image[0:height, :patch_size]  # DONE
    elif patch_size > width:
        canvas[:, 0: width] = image[:patch_size, 0:width]  # DONE
    elif x < 0:
        canvas[:, abs(x):patch_size] = image[0:patch_size, 0:patch_size]
    else:
        canvas[:, :] = image[0:patch_size, 0:patch_size]

    return canvas


def reorder_channels(mask, config):
    new_mask = np.zeros_like(mask)

    if config['classes'] == ["blood_vessels", "inflammations", "endocariums"]:
        new_mask = mask.numpy()
    elif config['classes'] == ["inflammations", "blood_vessels", "endocariums"]:
        new_mask[:, :, 0] = mask[:, :, 0]
        new_mask[:, :, 1] = mask[:, :, 2]
        new_mask[:, :, 2] = mask[:, :, 1]
        new_mask[:, :, 3] = mask[:, :, 3]

    elif config['classes'] == ["endocariums", "blood_vessels", "inflammations"]:
        new_mask[:, :, 0] = mask[:, :, 0]
        new_mask[:, :, 1] = mask[:, :, 2]
        new_mask[:, :, 2] = mask[:, :, 3]
        new_mask[:, :, 3] = mask[:, :, 1]

    return new_mask


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def deeplab_infere(networks, data, configs):
    patch_size = configs[0]['image_size']

    canvas = np.zeros_like(data, dtype='float16')
    divim = np.zeros_like(data, dtype='uint8')

    for (x, y, window) in sliding_window(data, stepSize=patch_size // 2, windowSize=(patch_size, patch_size)):
        if window.shape[0] != patch_size or window.shape[1] != patch_size:
            window = get_fragment(window, x, y, patch_size, fill_value=1)

        window = np.expand_dims(window, axis=0)
        window = tf.convert_to_tensor(window, dtype=tf.float32)
        final_mask = np.zeros((patch_size, patch_size, len(configs[0]['classes'])))
        masks = []
        # for idx, network in enumerate(networks):
        mask1 = networks[0].predict(window)
        mask2 = networks[1].predict(window)
        mask3 = networks[2].predict(window)

        mask1 = mask1[0]
        mask2 = mask2[0]
        mask3 = mask3[0]

        predicted1 = np.argmax(mask1, axis=-1)
        predicted2 = np.argmax(mask2, axis=-1)
        predicted3 = np.argmax(mask3, axis=-1)

        mask1 = tf.one_hot(predicted1, len(configs[0]['final_classes']))
        mask2 = tf.one_hot(predicted2, len(configs[0]['final_classes']))
        mask3 = tf.one_hot(predicted3, len(configs[0]['final_classes']))

        mask1 = reorder_channels(mask1, configs[0])[:, :, 1:]
        mask2 = reorder_channels(mask2, configs[1])[:, :, 1:]
        mask3 = reorder_channels(mask3, configs[2])[:, :, 1:]

        # masks.append(mask)

        final_mask[:, :, 0] = cv2.bitwise_or(mask3[:, :, 0], cv2.bitwise_or(mask2[:, :, 0], mask1[:, :, 0]))
        final_mask[:, :, 1] = cv2.bitwise_or(mask3[:, :, 1], cv2.bitwise_or(mask2[:, :, 1], mask1[:, :, 1]))
        final_mask[:, :, 2] = cv2.bitwise_or(mask3[:, :, 2], cv2.bitwise_or(mask2[:, :, 2], mask1[:, :, 2]))

        canvas[y:y + patch_size, x:x + patch_size] += final_mask[:patch_size +
                                                                 (canvas.shape[0] - y - final_mask.shape[0]), : patch_size + (canvas.shape[1] - x - final_mask.shape[1]), :]
        divim[y:y + patch_size, x:x + patch_size] += 1

    divim[divim == 0] = 1
    canvas /= divim

    # print(canvas.shape)
    # print(np.unique(canvas))

    return np.array(canvas > configs[0]['threshold'], dtype='uint8')


def custom_infere(network, data, config):
    if len(network) > 1:
        return deeplab_infere(network, data, config)


# def custom_infere(network, data, config):
#     patch_size = config['image_size']

#     canvas = np.zeros_like(data, dtype='float16')
#     divim = np.zeros_like(data, dtype='uint8')

#     for (x, y, window) in sliding_window(data, stepSize=patch_size // 2, windowSize=(patch_size, patch_size)):
#         if window.shape[0] != patch_size or window.shape[1] != patch_size:
#             window = get_fragment(window, x, y, patch_size, fill_value=1)

#         window = np.expand_dims(window, axis=0)
#         window = tf.convert_to_tensor(window, dtype=tf.float32)
#         mask = network.predict(window)
#         mask1 = mask[0]
#         predicted = np.argmax(mask1, axis=-1)

#         mask1 = tf.one_hot(predicted, len(config['final_classes']))
#         mask1 = reorder_channels(mask1, config)[:, :, 1:]

#         canvas[y:y + patch_size, x:x + patch_size] += mask1[:patch_size +
#                                                             (canvas.shape[0] - y - mask1.shape[0]), : patch_size + (canvas.shape[1] - x - mask1.shape[1]), :]
#         divim[y:y + patch_size, x:x + patch_size] += 1

#     divim[divim == 0] = 1
#     canvas /= divim

#     # print(canvas.shape)
#     # print(np.unique(canvas))

#     return np.array(canvas > config['threshold'], dtype='uint8')
