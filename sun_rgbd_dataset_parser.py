# parses the raw SUNRGBD dataset into a dataset that is usable for the project
import os


def image_parse(dir):
    # e.g. from files "img-0001.jpg" to files "1.jpg"
    for file in os.listdir(dir):
        new_name = file.replace('img-', '')
        new_name, ext = os.path.splitext(new_name)
        new_name = str(int(new_name)) + ext
        os.rename(os.path.join(dir, file), os.path.join(dir, new_name))
        print(file, ' --> ', new_name)


def labels_parse(dir, parent_dir):
    # label dir contains first all test images and then train images so we have to split them
    # create train & test labels directory if not exist
    if not os.path.exists(os.path.join(parent_dir, 'train_labels')):
        os.mkdir(os.path.join(parent_dir, 'train_labels'))
        os.mkdir(os.path.join(parent_dir, 'test_labels'))

    for file in os.listdir(dir):
        new_name = file.replace('img-', '')
        new_name, ext = os.path.splitext(new_name)
        int_name = int(new_name)

        # IF [the file is from train labels]
        if int_name > 5050:
            new_name = str(int_name - 5050) + ext
            new_name = os.path.join(parent_dir, 'train_labels', new_name)
        # The file is from test labels
        else:
            new_name = str(int_name) + ext
            new_name = os.path.join(parent_dir, 'test_labels', new_name)

        print(file, ' --> ', new_name)
        os.rename(os.path.join(dir, file), new_name)


print('\n\n############# Train images folder ##############')
image_parse('datasets/sun_rgbd/SUNRGBD-train_images')

print('\n\n############# Test images folder ###############')
image_parse('datasets/sun_rgbd/SUNRGBD-test_images')

print('\n\nDepth images are in good format')

print('\n\n############# Labels ####################')
labels_parse('datasets/sun_rgbd/sunrgbd_train_test_labels',
             'datasets/sun_rgbd')
