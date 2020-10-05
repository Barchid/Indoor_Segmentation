import os

nyu_v2 = os.path.join('datasets', 'nyu_v2_13c')
train_mask = os.path.join(nyu_v2, 'train_mask')
test_mask = os.path.join(nyu_v2, 'test_mask')


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def parse_mask_folder(dir):
    mkdir(train_mask)
    mkdir(test_mask)

    for file in os.listdir(dir):
        name, ext = os.path.splitext(file)
        name = name.replace("new_nyu_class13_", "")
        mask_id = int(name)
        new_name = str(mask_id) + ext

        new_path = os.path.join(dir, new_name)
        print(new_path)
        os.rename(os.path.join(dir, file), new_path)


if __name__ == "__main__":
    parse_mask_folder(train_mask)
    parse_mask_folder(test_mask)