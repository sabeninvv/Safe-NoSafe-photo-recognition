@cache(maxsize=None)
def crop2img2arr(path, in_shape=(128, 128), crop=False, center=False):
    try:
        img = Image.open(path, 'r')
        if crop:
            if img.size[0] >= img.size[1]:
                img = img.crop((0, 0, img.size[0], img.size[0]))
            else:
                img = img.crop((0, 0, img.size[1], img.size[1]))
        img = img.resize(in_shape)
        img = img.convert('RGB')
        if center:
            imgarr = np.array(img, dtype='float32')
            imgarr = imgarr - imgarr.mean()
            imgarr = imgarr / max(imgarr.max(), abs(imgarr.min()))
        else:
            imgarr = np.array(img, dtype='float32')
            imgarr /= 128.
            imgarr -= 1.
        return imgarr
    except:
        return None


def save2json(arr, name='post.json'):
    arr = np.expand_dims(arr, axis=0)
    with open(name, 'w') as file_object:
        json.dump({"signature_name": "serving_default",
                   "instances": arr.tolist()}, file_object)


def compare_imgs(names, main_dir):
    arr = []
    for name in names:
        path2img = os.path.join(main_dir, name)
        img2arr = crop2img2arr(path2img)
        if img2arr is not None:
            arr.append(img2arr)
    return np.array(arr)


def main():
    main_dir = os.getcwd()
    imgs = os.listdir(main_dir)
    arr = compare_imgs(imgs, main_dir)
    save2json(arr)


if __name__ == '__main__':
    main()