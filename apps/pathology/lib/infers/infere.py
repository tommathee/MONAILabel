import cv2 as cv
import numpy as np
from infere_config import InfereConfig
import argparse
import json
import tensorflow as tf
import albumentations as A


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
        canvas[0:height, 0:width] = image[:, :]  # DONE
    elif x < 0 and patch_size > height:
        canvas[0:height, abs(x):patch_size] = image[0:height, 0:patch_size]
    elif patch_size > height:
        canvas[0: height, :] = image[:, :]  # DONE
    elif patch_size > width:
        canvas[:, 0: width] = image[:, :]  # DONE
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


def applicate_augmentations(aug, img, mask, batch_size):
    output_img = []
    for image_idx in range(len(img)):
        new_img = np.zeros_like(img[image_idx])
        for idx_batch in range(batch_size):
            augmented = aug(image=img[image_idx][idx_batch], mask=mask[idx_batch])
            new_img[idx_batch] = augmented['image']
        output_img.append(new_img)
    return output_img, mask


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
            window = get_fragment(window, x, y, patch_size, fill_value=1.)

        # convert bgr to rgb
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

        final_mask[:, :, 0] = cv.bitwise_or(mask3[:, :, 0], cv.bitwise_or(mask2[:, :, 0], mask1[:, :, 0]))
        final_mask[:, :, 1] = cv.bitwise_or(mask3[:, :, 1], cv.bitwise_or(mask2[:, :, 1], mask1[:, :, 1]))
        final_mask[:, :, 2] = cv.bitwise_or(mask3[:, :, 2], cv.bitwise_or(mask2[:, :, 2], mask1[:, :, 2]))

        canvas[y:y + patch_size, x:x + patch_size] += final_mask[:patch_size +
                                                                 (canvas.shape[0] - y - final_mask.shape[0]), : patch_size + (canvas.shape[1] - x - final_mask.shape[1]), :]
        divim[y:y + patch_size, x:x + patch_size] += 1

    divim[divim == 0] = 1
    canvas /= divim

    return np.array(canvas > configs[0]['threshold'], dtype='uint8')


def infere(network, data, config):
    patch_size = config['image_size']
    horizontal_flip = A.HorizontalFlip(p=1)
    vertical_flip = A.VerticalFlip(p=1)

    hor_ver_flip = A.Compose([
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1)
    ])

    canvas = np.zeros_like(data, dtype='float16')
    divim = np.zeros_like(data, dtype='uint8')

    mask = np.zeros((1, patch_size, patch_size, len(config['classes'])))

    for (x, y, window) in sliding_window(data, stepSize=patch_size // 2, windowSize=(patch_size, patch_size)):
        if window.shape[0] != patch_size or window.shape[1] != patch_size:
            window = get_fragment(window, x, y, patch_size, fill_value=1)

        window2 = tf.image.resize(window, [patch_size // 2, patch_size // 2], method='bilinear')
        # Pre ucely nahradenia segmentacie jadierok
        window = np.concatenate(
            (window, np.zeros((window.shape[0], window.shape[1], 1))),
            axis=-1
        )
        window = np.expand_dims(window, axis=0)
        window2 = np.expand_dims(window2, axis=0)
        #window = tf.convert_to_tensor(window, dtype=tf.float32)
        img = [window, window2]

        pred_mask = network.predict(img)

        img1, _ = applicate_augmentations(horizontal_flip, img, mask, 1)
        pred_mask1 = network.predict(img1)

        img2, _ = applicate_augmentations(vertical_flip, img, mask, 1)
        pred_mask2 = network.predict(img2)

        img3, _ = applicate_augmentations(hor_ver_flip, img, mask, 1)
        pred_mask3 = network.predict(img3)

        pred_mask1, _ = applicate_augmentations(horizontal_flip, pred_mask1, mask, 1)
        pred_mask2, _ = applicate_augmentations(vertical_flip, pred_mask2, mask, 1)
        pred_mask3, _ = applicate_augmentations(hor_ver_flip, pred_mask3, mask, 1)

        final_mask = np.sum([pred_mask, pred_mask1, pred_mask2, pred_mask3], axis=0) / 4

        for idx in range(pred_mask.shape[0]):
            final_mask_idx = final_mask[idx]
            canvas[y:y + patch_size, x:x + patch_size] += final_mask_idx[
                : patch_size + (canvas.shape[0] - y - final_mask_idx.shape[0]),
                : patch_size + (canvas.shape[1] - x - final_mask_idx.shape[1]),
                :
            ]
            divim[y:y + patch_size, x:x + patch_size] += 1

    divim[divim == 0] = 1
    canvas /= divim

    return np.array(canvas > config['threshold'], dtype='uint8')


def main(directory='tmp'):
    data = np.load(f'{directory}/data.npy', mmap_mode='r')
    with open(f'{directory}/configs.json') as f:
        configs = json.load(f)
    with open(f'{directory}/paths.json') as f:
        paths = json.load(f)

    networks = []
    for idx, config in enumerate(configs):
        model = InfereConfig().get_model(config['model'])
        networks.append(model(config).create_model())
        networks[idx].load_weights(paths[idx].replace('.index', ''))

    if len(networks) > 1:
        mask = deeplab_infere(networks, data, configs)
    else:
        mask = infere(networks[0], data, configs[0])

    np.save(f'{directory}/mask.npy', mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='tmp')
    args = parser.parse_args()

    main(args.directory)
