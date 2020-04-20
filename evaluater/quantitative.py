"""Evaluates the class-wise IoU, mean IoU, pixel accuracy and class-wise mean pixel accuracy for the test set"""
import numpy as np


def get_iou(pred, gt, num_class):
    if pred.shape != gt.shape:
        print('pred shape', pred.shape, 'gt shape', gt.shape)
    assert (pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    max_label = int(num_class) - 1  # labels from 0,1, ... C
    count = np.zeros((max_label + 1,))
    for j in range(1, max_label + 1):
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:]) / float(len(np.unique(gt))-1)
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
    num_correct_lbl, num_samples_lbl = 0, 0
    Aiou = 0
    Am_acc = 0
    pixel_acc = 0

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

        # retrieve pixel accuracy for the current image
        pixel_acc += pixel_accuracy(pred, gt)

        # update means
        Aiou += iou
        class_Aiou += class_iou
        count_class[class_ct] += 1

    # mean of all pixel accuracies
    pixel_acc = pixel_acc / len(datagen)

    for i, (img, lbl) in enumerate(loader):
        img_var = Variable(img.cuda(), volatile=True)
        _, _, _, _, _, pred_labels = model(img_var)

        _, preds_lbl = pred_labels.data.cpu().max(1)
        lbl = np.squeeze(lbl).numpy()
        preds_lbl = np.squeeze(preds_lbl).numpy()
        preds_lbl = misc.imresize(preds_lbl, lbl.shape, mode='F')

        img = img.cpu().numpy()
        img = img[0].transpose(1, 2, 0)
        iou, class_iou, class_ct = get_iou(preds_lbl, lbl)
        Aiou += iou
        class_Aiou += class_iou
        count_class[class_ct] += 1
        print("processing: %d/%d" % (i, len(loader)))

        lbl_0 = (lbl == 0)
        preds_lbl[lbl_0] = 0
        # save images
        # plt.imsave(os.path.join(out_prdlbl,"%d.png"%(i+1)),preds_lbl*4,cmap='nipy_spectral',vmin=0,vmax=(num_class-1)*4)

        t_cls = np.unique(lbl)
        if t_cls[0] == 0:
            t_cls = t_cls[1:]

        mask_lbl = (lbl != 0)
        lbl = torch.from_numpy(lbl[mask_lbl]).long()
        preds_lbl = torch.from_numpy(preds_lbl[mask_lbl]).long()
        num_correct_lbl += (preds_lbl.long() == lbl.long()).sum()
        num_samples_lbl += preds_lbl.numel()

        m_acc = 0
        for cls in t_cls:
            m_acc += float((preds_lbl[lbl.long() == cls].long()
                            == cls).sum())/float((lbl.long() == cls).sum())
        Am_acc += m_acc/len(t_cls)

    acc_lbl = float(num_correct_lbl)/num_samples_lbl
    Aiou /= num_samples
    Am_acc /= num_samples
    class_Aiou[1:] /= count_class[1:]
    class_Aiou = np.where(np.isnan(class_Aiou), 0, class_Aiou)

    return Aiou, Am_acc, acc_lbl, class_Aiou
