import os
import cv2
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import torch.utils.data as data


class VOCSegmentation(data.Dataset):
    def __init__(self, voc_root, transforms=None, aug=False, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()

        assert os.path.exists(voc_root), "path '{}' does not exist.".format(voc_root)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')
        txt_path = os.path.join(voc_root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)

        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms
        # data augumentation
        self.aug = aug
        self.seq = iaa.Sequential([
                    iaa.Affine( 
                            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, 
                            translate_percent={"x": (-0.10, 0.10), "y": (-0.10, 0.10)},   
                            shear=(-8, 8),  
                            order=[0, 1],  
                            cval=0,
                            mode="constant" 
                            ),
                    iaa.SomeOf((0, 5),
                            [iaa.GaussianBlur(sigma=(1, 3.0)), 
                            iaa.Sharpen(alpha=1.0,lightness=(0.5, 1.5)),
                            iaa.Dropout([0, 0.05]),
                            iaa.LinearContrast((0.5, 1.5)),
                            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)
                            ])
                    ])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        
        if self.aug:
            img = cv2.imread(self.images[index])
            target = cv2.imread(self.masks[index])
            segmap = SegmentationMapsOnImage(target, shape=img.shape)

            images_aug, segmaps_aug = self.seq(image=img, segmentation_maps=segmap)
            # images_aug, segmaps_aug = self.seq_2(image=images_aug, segmentation_maps=segmaps_aug)
            segmaps_aug = segmaps_aug.arr.astype(np.uint8)
            img = Image.fromarray(cv2.cvtColor(images_aug, cv2.COLOR_BGR2RGB))
            target = Image.fromarray(segmaps_aug).convert("L")
            # target = Image.fromarray(cv2.cvtColor(segmaps_aug, cv2.COLOR_BGR2RGB)).convert("L")
            # target = Image.fromarray(segmaps_aug)
            if self.transforms is not None:
                img, target = self.transforms(img, target)
        else:
            img = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.masks[index]).convert('L')
            if self.transforms is not None:
                img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
