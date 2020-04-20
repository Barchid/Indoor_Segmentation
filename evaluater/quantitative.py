"""Evaluates the class-wise IoU, mean IoU, pixel accuracy and class-wise mean pixel accuracy for the test set"""
import numpy as np
from data_generators.segmentation_data_generator import SegmentationDataGenerator

def get_iou(pred, gt, num_class):
    if pred.shape != gt.shape:
        print('pred shape', pred.shape, 'gt shape', gt.shape)
    assert (pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    count = np.zeros((num_class,))
    for j in range(0, num_class):
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:]) / float(len(np.unique(gt)))
    count_class = (result_class != 0)

    return Aiou, result_class, count_class


def onehot_to_argmax(pred, gt):
    """Converts the one-hot encoded labels to the argmax mask for prediction and ground truth images
    """
    pred = np.argmax(pred, axis=-1).astype(np.uint8)
    gt = np.argmax(gt, axis=-1).astype(np.uint8)
    return pred, gt


def pixel_accuracy(pred, gt):
    """Computes the pixel accuracy between the prediction and the ground truth masks
    """
    correct_pred_count = (pred == gt).sum()
    total_pixels = gt.shape[0] * gt.shape[1]

    return correct_pred_count / total_pixels


def mean_pixel_accuracy(pred, gt):
    """Computes the mean pixel accuracy between the prediction and the ground truth masks
    """
    # the notation used is from https://github.com/martinkersner/py_img_seg_eval
    mean_pixel_acc = 0.

    cls = np.unique(gt)  # classes included in ground truth
    n_cl = float(len(cls))  # n_cl = number of class in ground truth

    for i in cls:
        # t_i = total number of pixel of class i in ground truth
        t_i = (gt == i).sum()

        # n_ij = number of pixels of class i predicted as class j
        class_pred = (pred == i).astype(np.uint8)
        class_gt = (gt == i).astype(np.uint8)
        # n_ii = number of correct prediction
        n_ii = (class_pred == class_gt).sum()

        mean_pixel_acc += n_ii / t_i

    mean_pixel_acc = (1./n_cl) * mean_pixel_acc

    return mean_pixel_acc


def evaluate_accuracy(model, config):
    num_class = config.model.classes
    Aiou = 0
    pixel_acc = 0.
    mean_pixel_acc = 0.

    # Data generator creation
    datagen = SegmentationDataGenerator(config, is_training_set=False)

    num_samples = len(datagen)

    class_Aiou = np.zeros((num_class,))
    count_class = np.zeros((num_class,))

    out_prdlbl = 'outimgs/'

    # FOR EACH [image] IN [data generator]
    for i in range(len(datagen)):
        # retrieve
        X, Y = datagen[i]
        gt = Y[0]
        pred = model.predict(
            X, batch_size=config.trainer.batch_size, verbose=1)
        pred = pred[0]

        pred, gt = onehot_to_argmax(pred, gt)

        # retrieve IoUs
        iou, class_iou, class_ct = get_iou(pred, gt, num_class)

        # retrieve pixel accuracy for the current prediction
        pixel_acc += pixel_accuracy(pred, gt)

        # retrieve mean pixel accuracy for the current prediction
        mean_pixel_acc += mean_pixel_accuracy(pred, gt)

        # update IoUs stats
        Aiou += iou
        class_Aiou += class_iou
        count_class[class_ct] += 1

    # mean of all pixel accuracies
    pixel_acc = pixel_acc / len(datagen)

    # mean of all mean pixel accuracies
    mean_pixel_acc = mean_pixel_acc / len(datagen)

    # finish equations for IoUs
    Aiou /= num_samples
    class_Aiou[:] /= count_class[:]
    # remove div by 0 and replace them by 0
    class_Aiou = np.where(np.isnan(class_Aiou), 0, class_Aiou)

    return Aiou, class_Aiou, pixel_acc, mean_pixel_acc
