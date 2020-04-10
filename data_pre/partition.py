import numpy as np
import SimpleITK as sitk


def partition(option_p, patch_size,overlap_size, mode=None, img_sz=(-1,-1,-1), flicker_on=False, flicker_mode='rand'):
    padding_mode = option_p[('padding_mode', 'reflect', 'padding_mode')]
    mode = mode if mode is not None else option_p[('mode', 'pred', 'eval or pred')]
    flicker_range = option_p[('flicker_range', 0, 'flicker range')]
    partition = Partition(patch_size, overlap_size, padding_mode=padding_mode, mode=mode,img_sz=img_sz, flicker_on=flicker_on,flicker_range=flicker_range, flicker_mode=flicker_mode)
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

    def __init__(self, tile_size, overlap_size, padding_mode='reflect', mode="pred", img_sz=None,flicker_on=False,flicker_range=0,flicker_mode='rand'):
        self.tile_size = np.flipud(np.asarray(tile_size))  # convert the itk coord to np coord
        self.overlap_size = np.flipud(np.asarray(overlap_size)) # convert the itk coord to np coord
        self.image_size = img_sz
        self.padding_mode = padding_mode
        self.mode = mode
        self.flicker_on=flicker_on
        self.flicker_range = flicker_range
        self.flicker_mode= flicker_mode



    def __call__(self, sample,disp=0):
        """
        :param image: (simpleITK image) 3D Image to be partitioned
        :param seg: (simpleITK image) 3D segmentation label mask to be partitioned
        :return: N partitioned image and label patches
            {'image':  Nx1xDxHxW, 'label':  Nx1xDxHxW }
        """
        # get numpy array from simpleITK images
        images_t = sample['image']

        #self.image = sample['image']
        is_numpy = False
        if not isinstance(images_t,list):
            # is not list, then it should be itk image
            images = [sitk.GetArrayFromImage(images_t)]
        else:
            if not isinstance(images_t[0], np.ndarray):
                images = [ sitk.GetArrayFromImage(image) for image in images_t]
            else:
                is_numpy = True
                images = images_t
        if 'label' in sample:
            if not is_numpy:
                seg_np = sitk.GetArrayFromImage(sample['label'])
            else:
                seg_np = sample['label']

        self.image_size = np.array(images[0].shape) # np coord
        self.effective_size = self.tile_size - self.overlap_size * 2  # size effective region of tiles after cropping
        self.tiles_grid_size = np.ceil(self.image_size / self.effective_size).astype(int)  # size of tiles grid
        self.padded_size = self.effective_size * self.tiles_grid_size + self.overlap_size * 2 - self.image_size  # size difference of padded image with original image
        #print("partitioning, the padded size is {}".format(self.padded_size))
        if self.flicker_on:
            pp = self.flicker_range
        else:
            pp=0

        if self.mode == 'eval':
            seg_padded = np.pad(seg_np,
                                pad_width=((self.overlap_size[0]+pp, self.padded_size[0] - self.overlap_size[0]+pp),
                                           (self.overlap_size[1]+pp, self.padded_size[1] - self.overlap_size[1]+pp),
                                           (self.overlap_size[2]+pp, self.padded_size[2] - self.overlap_size[2]+pp)),
                                mode=self.padding_mode)

        image_tile_list = []
        start_coord_list = []
        seg_tile_list = []
        image_padded_list = [np.pad(image_np,
                              pad_width=((self.overlap_size[0] + pp, self.padded_size[0] - self.overlap_size[0] + pp),
                                         (self.overlap_size[1] + pp, self.padded_size[1] - self.overlap_size[1] + pp),
                                         (self.overlap_size[2] + pp, self.padded_size[2] - self.overlap_size[2] + pp)),
                              mode=self.padding_mode) for image_np in images]

        for i in range(self.tiles_grid_size[0]):
            for j in range(self.tiles_grid_size[1]):
                for k in range(self.tiles_grid_size[2]):
                    if self.flicker_on:
                        if self.flicker_mode=='rand':
                            ri, rj, rk = [np.random.randint(-self.flicker_range,self.flicker_range) for _ in range(3)]
                        elif self.flicker_mode =='ensemble':
                            ri,rj,rk = disp
                    else:
                        ri,rj,rk= 0, 0, 0
                    image_tile_temp_list = [image_padded[
                                      i * self.effective_size[0]+ri+pp:i * self.effective_size[0] + self.tile_size[0]+ri+pp,
                                      j * self.effective_size[1]+rj+pp:j * self.effective_size[1] + self.tile_size[1]+rj+pp,
                                      k * self.effective_size[2]+rk+pp:k * self.effective_size[2] + self.tile_size[2]+rk+pp]
                                            for image_padded in image_padded_list]
                    image_tile_list.append(np.stack(image_tile_temp_list,0))
                    start_coord_list.append((i * self.effective_size[0]+ri,j * self.effective_size[1]+rj,k * self.effective_size[2]+rk))

                    if self.mode == 'eval':
                        seg_tile_temp = seg_padded[
                                        i * self.effective_size[0]+ri+pp:i * self.effective_size[0] + self.tile_size[0]+ri+pp,
                                        j * self.effective_size[1]+rj+pp:j * self.effective_size[1] + self.tile_size[1]+rj+pp,
                                        k * self.effective_size[2]+rk+pp:k * self.effective_size[2] + self.tile_size[2]+rk+pp]
                        seg_tile_list.append(np.expand_dims(seg_tile_temp, axis=0))

        # sample['image'] = np.stack(image_tile_list, 0)
        # sample['segmentation'] = np.stack(seg_tile_list, 0)
        trans_sample ={}

        trans_sample['image'] = np.stack(image_tile_list, 0) # N*C*xyz
        if 'label'in sample:
            if self.mode == 'pred':
                trans_sample['label'] = np.expand_dims(np.expand_dims(seg_np, axis=0), axis=0)  #1*XYZ
            else:
                trans_sample['label'] = np.stack(seg_tile_list, 0)  # N*1*xyz
        trans_sample['tile_size'] = self.tile_size
        trans_sample['overlap_size'] = self.overlap_size
        trans_sample['padding_mode'] = self.padding_mode
        trans_sample['flicker_on'] = self.flicker_on
        trans_sample['disp'] = disp
        trans_sample['num_crops_per_img'] = len(image_tile_list)
        trans_sample['start_coord_list'] = start_coord_list

        return trans_sample

    def assemble(self, tiles,image_size=None, is_vote=False):
        """
        Assembles segmentation of small patches into the original size
        :param tiles: Nxhxdxw tensor contains N small patches of size hxdxw
        :param is_vote:
        :return: a segmentation information

        """

        if image_size is not None:
            self.image_size = image_size
        self.effective_size = self.tile_size - self.overlap_size * 2  # size effective region of tiles after cropping
        self.tiles_grid_size = np.ceil(self.image_size / self.effective_size).astype(int)  # size of tiles grid
        self.padded_size = self.effective_size * self.tiles_grid_size + self.overlap_size * 2 - self.image_size  # size difference of padded image with original image

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

        # seg_image = sitk.GetImageFromArray(seg_reassemble)
        # seg_image.CopyInformation(self.image)
        seg_reassemble = np.expand_dims(np.expand_dims(seg_reassemble, axis=0), axis=0)
        return seg_reassemble



    def assemble_multi_torch(self, tiles,image_size=None):
        """
        Assembles segmentation of small patches into the original size
        :param tiles: Nxhxdxw tensor contains N small patches of size hxdxw
        :param is_vote:
        :return: a segmentation information

        """
        import torch
        if image_size is not None:
            self.image_size = image_size
        self.effective_size = self.tile_size - self.overlap_size * 2  # size effective region of tiles after cropping
        self.tiles_grid_size = np.ceil(self.image_size / self.effective_size).astype(int)  # size of tiles grid
        self.padded_size = self.effective_size * self.tiles_grid_size + self.overlap_size * 2 - self.image_size  # size difference of padded image with original image


        seg_reassemble = torch.zeros([1,tiles.shape[1]]+list(self.effective_size * self.tiles_grid_size)).to(tiles.device)
        for i in range(self.tiles_grid_size[0]):
            for j in range(self.tiles_grid_size[1]):
                for k in range(self.tiles_grid_size[2]):
                    ind = i * self.tiles_grid_size[1] * self.tiles_grid_size[2] + j * self.tiles_grid_size[2] + k
                    seg_reassemble[0,:,i * self.effective_size[0]:(i + 1) * self.effective_size[0],
                    j * self.effective_size[1]:(j + 1) * self.effective_size[1],
                    k * self.effective_size[2]:(k + 1) * self.effective_size[2]] = \
                        tiles[ind][:,self.overlap_size[0]:self.tile_size[0] - self.overlap_size[0],
                        self.overlap_size[1]:self.tile_size[1] - self.overlap_size[1],
                        self.overlap_size[2]:self.tile_size[2] - self.overlap_size[2]]
        seg_reassemble = seg_reassemble[:,:,:self.image_size[0], :self.image_size[1], :self.image_size[2]]

        # seg_image = sitk.GetImageFromArray(seg_reassemble)
        # seg_image.CopyInformation(self.image)
        return seg_reassemble

