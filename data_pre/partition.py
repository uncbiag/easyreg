import numpy as np
import SimpleITK as sitk


def partition(option_p):
    overlap_size = option_p[('overlap_size',tuple([16,16,8]), 'overlap_size')]
    padding_mode = option_p[('padding_mode', 'reflect', 'padding_mode')]
    mode = option_p[('mode', 'eval', 'eval or pred')]
    patch_size = option_p[('patch_size', [128, 128, 32], 'patch size')]
    partition = Partition(patch_size, overlap_size, padding_mode=padding_mode, mode=mode)
    return partition




class Partition(object):
    """partition a 3D volume into small 3D patches using the overlap tiling strategy described in paper:
    "U-net: Convolutional networks for biomedical image segmentation." by Ronneberger, Olaf, Philipp Fischer,
    and Thomas Brox. In International Conference on Medical Image Computing and Computer-Assisted Intervention,
    pp. 234-241. Springer, Cham, 2015.

    Note: BE CAREFUL about the order of dimensions for image:
            The simpleITK image are in order x, y, z
            The numpy array/torch tensor are in order z, y, x


    :param tile_size (tuple of 3 or 1x3 np array): size of partitioned patches
    :param self.overlap_size (tuple of 3 or 1x3 np array): the size of overlapping region at both end of each dimension
    :param padding_mode (tuple of 3 or 1x3 np array): the mode of numpy.pad when padding extra region for image
    :param mode: "pred": only image is partitioned; "eval": both image and segmentation are partitioned TODO
    """

    def __init__(self, tile_size, overlap_size, padding_mode='reflect', mode="pred"):
        self.tile_size = np.flipud(
            np.asarray(tile_size))  # flip the size order to match the numpy array(check the note)
        self.overlap_size = np.flipud(np.asarray(overlap_size))
        self.padding_mode = padding_mode
        self.mode = mode



    def __call__(self, sample):
        """
        :param image: (simpleITK image) 3D Image to be partitioned
        :param seg: (simpleITK image) 3D segmentation label mask to be partitioned
        :return: N partitioned image and label patches
            {'img':  Nx1xDxHxW, 'label':  Nx1xDxHxW }
        """
        # get numpy array from simpleITK images
        image_np = sitk.GetArrayFromImage(sample['img'])
        seg_np = sitk.GetArrayFromImage(sample['seg'])
        self.image = sample['img']
        self.image_size = np.array(image_np.shape)  # size of input image
        self.effective_size = self.tile_size - self.overlap_size * 2  # size effective region of tiles after cropping
        self.tiles_grid_size = np.ceil(self.image_size / self.effective_size).astype(int)  # size of tiles grid
        self.padded_size = self.effective_size * self.tiles_grid_size + self.overlap_size * 2 - self.image_size  # size difference of padded image with original image

        image_padded = np.pad(image_np,
                              pad_width=((self.overlap_size[0], self.padded_size[0] - self.overlap_size[0]),
                                         (self.overlap_size[1], self.padded_size[1] - self.overlap_size[1]),
                                         (self.overlap_size[2], self.padded_size[2] - self.overlap_size[2])),
                              mode=self.padding_mode)

        if self.mode == 'eval':
            seg_padded = np.pad(seg_np,
                                pad_width=((self.overlap_size[0], self.padded_size[0] - self.overlap_size[0]),
                                           (self.overlap_size[1], self.padded_size[1] - self.overlap_size[1]),
                                           (self.overlap_size[2], self.padded_size[2] - self.overlap_size[2])),
                                mode=self.padding_mode)

        image_tile_list = []
        seg_tile_list = []
        for i in range(self.tiles_grid_size[0]):
            for j in range(self.tiles_grid_size[1]):
                for k in range(self.tiles_grid_size[2]):
                    image_tile_temp = image_padded[
                                      i * self.effective_size[0]:i * self.effective_size[0] + self.tile_size[0],
                                      j * self.effective_size[1]:j * self.effective_size[1] + self.tile_size[1],
                                      k * self.effective_size[2]:k * self.effective_size[2] + self.tile_size[2]]
                    image_tile_list.append(image_tile_temp)

                    if self.mode == 'eval':
                        seg_tile_temp = seg_padded[
                                        i * self.effective_size[0]:i * self.effective_size[0] + self.tile_size[0],
                                        j * self.effective_size[1]:j * self.effective_size[1] + self.tile_size[1],
                                        k * self.effective_size[2]:k * self.effective_size[2] + self.tile_size[2]]
                        seg_tile_list.append(seg_tile_temp)

        # sample['img'] = np.stack(image_tile_list, 0)
        # sample['segmentation'] = np.stack(seg_tile_list, 0)
        trans_sample ={}

        trans_sample['img'] = np.expand_dims(np.stack(image_tile_list, 0), axis=1)
        if self.mode == 'pred':
            trans_sample['seg'] = np.expand_dims(seg_np, axis=0)
        else:
            trans_sample['seg'] = np.expand_dims(np.stack(seg_tile_list, 0), axis=1)
        trans_sample['effective_size'] = self.effective_size

        return trans_sample

    def assemble(self, tiles, is_vote=False):
        """
        Assembles segmentation of small patches into the original size
        :param tiles: Nxhxdxw tensor contains N small patches of size hxdxw
        :param is_vote:
        :return: a segmentation information
        """
        tiles = tiles.numpy()

        if is_vote:
            label_class = np.unique(tiles)
            seg_vote_array = np.zeros(
                np.insert(self.effective_size * self.tiles_grid_size + self.overlap_size * 2, 0, label_class.size),
                dtype=int)
            for i in range(self.tiles_grid_size[0]):
                for j in range(self.tiles_grid_size[1]):
                    for k in range(self.tiles_grid_size[2]):
                        ind = i * self.tiles_grid_size[1] * self.tiles_grid_size[2] + j * self.tiles_grid_size[2] + k
                        for label in label_class:
                            local_ind = np.where(
                                tiles[ind] == label)  # get the coordinates in local patch of each class
                            global_ind = (local_ind[0] + i * self.effective_size[0],
                                          local_ind[1] + j * self.effective_size[1],
                                          local_ind[2] + k * self.effective_size[2])  # transfer into global coordinates
                            seg_vote_array[label][global_ind] += 1  # vote for each glass

            seg_reassemble = np.argmax(seg_vote_array, axis=0)[
                             self.overlap_size[0]:self.overlap_size[0] + self.image_size[0],
                             self.overlap_size[1]:self.overlap_size[1] + self.image_size[1],
                             self.overlap_size[2]:self.overlap_size[2] + self.image_size[2]].astype(np.uint8)

            pass

        else:
            seg_reassemble = np.zeros(self.effective_size * self.tiles_grid_size)
            for i in range(self.tiles_grid_size[0]):
                for j in range(self.tiles_grid_size[1]):
                    for k in range(self.tiles_grid_size[2]):
                        ind = i * self.tiles_grid_size[1] * self.tiles_grid_size[2] + j * self.tiles_grid_size[2] + k
                        seg_reassemble[i * self.effective_size[0]:(i + 1) * self.effective_size[0],
                        j * self.effective_size[1]:(j + 1) * self.effective_size[1],
                        k * self.effective_size[2]:(k + 1) * self.effective_size[2]] = \
                            tiles[ind][self.overlap_size[0]:self.tile_size[0] - self.overlap_size[0],
                            self.overlap_size[1]:self.tile_size[1] - self.overlap_size[1],
                            self.overlap_size[2]:self.tile_size[2] - self.overlap_size[2]]
            seg_reassemble = seg_reassemble[:self.image_size[0], :self.image_size[1], :self.image_size[2]]

        seg_image = sitk.GetImageFromArray(seg_reassemble)
        seg_image.CopyInformation(self.image)

        return seg_image
