import os

nyu_v2 = os.path.join('datasets', 'nyu_v2')
train_rgb = os.path.join(nyu_v2, 'train_rgb')
test_rgb = os.path.join(nyu_v2, 'test_rgb')
train_mask = os.path.join(nyu_v2, 'train_mask')
test_mask = os.path.join(nyu_v2, 'test_mask')
train_depth = os.path.join(nyu_v2, 'train_depth')
test_depth = os.path.join(nyu_v2, 'test_depth')


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def rename_directories_if_needed():
    """Rename the directories if they still have the old names (from the downloaded zip file).
    """
    old_test_rgb = os.path.join(nyu_v2, 'nyu_test_rgb')
    if os.path.exists(old_test_rgb):
        os.rename(old_test_rgb, test_rgb)

    old_train_rgb = os.path.join(nyu_v2, 'nyu_train_rgb')
    if os.path.exists(old_train_rgb):
        os.rename(old_train_rgb, train_rgb)


def parse_image_folder(dir):
    # e.g. from files "nyu_rgb_0001.jpg" to files "1.jpg"
    for file in os.listdir(dir):
        new_name = file.replace('nyu_rgb_', '')
        new_name, ext = os.path.splitext(new_name)
        new_name = str(int(new_name)) + ext
        os.rename(os.path.join(dir, file), os.path.join(dir, new_name))
        print(file, ' --> ', new_name)


def parse_depth_folder(dir):
    # e.g. from files "depth_0001.jpg" to files "1.jpg"
    for file in os.listdir(dir):
        new_name = file.replace('depth_', '')
        new_name, ext = os.path.splitext(new_name)
        new_name = str(int(new_name)) + ext
        os.rename(os.path.join(dir, file), os.path.join(dir, new_name))
        print(file, ' --> ', new_name)


def parse_mask_folder(dir):
    mkdir(train_mask)
    mkdir(test_mask)

    for file in os.listdir(dir):
        name, ext = os.path.splitext(file)
        mask_id = int(name) + 1
        new_name = str(mask_id) + ext

        # check if the mask image is in test or train split
        # IF the related RGB image is in the train set, then move mask to the train mask dir
        if os.path.exists(os.path.join(train_rgb, new_name)):
            os.rename(os.path.join(dir, file),
                      os.path.join(train_mask, new_name))
        else:
            os.rename(os.path.join(dir, file),
                      os.path.join(test_mask, new_name))


if __name__ == "__main__":
    rename_directories_if_needed()
    parse_image_folder(train_rgb)
    parse_image_folder(test_rgb)
    parse_depth_folder(train_depth)
    parse_depth_folder(test_depth)
    parse_mask_folder(os.path.join(nyu_v2, 'labels_40'))
