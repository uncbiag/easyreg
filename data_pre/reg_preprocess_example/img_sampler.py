"""
image sampler code is clone from
https://raw.githubusercontent.com/acil-bwh/ChestImagingPlatform/develop/cip_python/dcnn/data/data_processing.py
"""
import math
import vtk
import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import scipy.ndimage.interpolation as scipy_interpolation


class DataProcessing(object):
    @classmethod
    def resample_image_itk(
        cls, image, output_size, output_type=None, interpolator=sitk.sitkBSpline
    ):
        """
        Image resampling using ITK
        :param image: simpleITK image
        :param output_size: numpy array or tuple. Output size
        :param output_type: simpleITK output data type. If None, use the same as 'image'
        :param interpolator: simpleITK interpolator (default: BSpline)
        :return: tuple with simpleITK image and array with the resulting output spacing
        """
        if not isinstance(output_size, np.ndarray):
            output_size = np.array(output_size)
        factor = np.asarray(image.GetSize()) / output_size.astype(np.float32)
        output_spacing = np.asarray(image.GetSpacing()) * factor

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetSize(output_size.tolist())
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputSpacing(output_spacing)
        resampler.SetOutputPixelType(
            output_type if output_type is not None else image.GetPixelIDValue()
        )
        resampler.SetOutputOrigin(image.GetOrigin())
        return resampler.Execute(image), output_spacing

    @classmethod
    def resample_image_itk_by_spacing(
        cls, image, output_spacing, output_type=None, interpolator=sitk.sitkBSpline
    ):
        """
        Image resampling using ITK
        :param image: simpleITK image
        :param output_spacing: numpy array or tuple. Output spacing
        :param output_type: simpleITK output data type. If None, use the same as 'image'
        :param interpolator: simpleITK interpolator (default: BSpline)
        :return: tuple with simpleITK image and array with the resulting output spacing
        """
        if not isinstance(output_spacing, np.ndarray):
            output_spacing = np.array(output_spacing)
        factor = np.asarray(image.GetSpacing()) / output_spacing.astype(np.float32)
        output_size = np.round(np.asarray(image.GetSize()) * factor + 0.0005).astype(
            np.uint32
        )

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetSize(output_size.tolist())
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputSpacing(output_spacing)
        resampler.SetOutputPixelType(
            output_type if output_type is not None else image.GetPixelIDValue()
        )
        resampler.SetOutputOrigin(image.GetOrigin())
        return resampler.Execute(image)

    @classmethod
    def resample_image_itk_by_spacing_and_size(
        cls,
        image,
        output_spacing,
        output_size,
        output_type=None,
        interpolator=sitk.sitkBSpline,
        padding_value=-1024,
        center_padding=True,
    ):
        """
        Image resampling using ITK
        :param image: simpleITK image
        :param output_spacing: numpy array or tuple. Output spacing
        :param output_size: numpy array or tuple. Output size
        :param output_type: simpleITK output data type. If None, use the same as 'image'
        :param interpolator: simpleITK interpolator (default: BSpline)
        :param padding_value: pixel padding value when a transformed pixel is outside of the image
        :return: tuple with simpleITK image and array with the resulting output spacing
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetSize(output_size)
        resampler.SetDefaultPixelValue(padding_value)
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputSpacing(np.array(output_spacing))
        resampler.SetOutputPixelType(
            output_type if output_type is not None else image.GetPixelIDValue()
        )
        factor = np.asarray(image.GetSpacing()) / np.asarray(output_spacing).astype(
            np.float32
        )

        # Get new output origin
        if center_padding:
            real_output_size = np.round(
                np.asarray(image.GetSize()) * factor + 0.0005
            ).astype(np.uint32)
            diff = ((output_size - real_output_size) * np.asarray(output_spacing)) / 2
            output_origin = np.asarray(image.GetOrigin()) - diff
        else:
            output_origin = np.asarray(image.GetOrigin())

        resampler.SetOutputOrigin(output_origin)
        return resampler.Execute(image)

    @classmethod
    def reslice_3D_image_vtk(
        cls, image, x_axis, y_axis, z_axis, center_point, target_size, output_spacing
    ):
        """
        3D image reslicing using vtk.
        :param image: VTK image
        :param x_axis: tuple. X direction for vtk SetResliceAxesDirectionCosines function used to specify the orientation of the slice. The direction cosines give the x, y, and z axes for the output volume
        :param y_axis: tuple. Y direction for vtk SetResliceAxesDirectionCosines function used to specify the orientation of the slice. The direction cosines give the x, y, and z axes for the output volume
        :param z_axis: tuple. Z direction for vtk SetResliceAxesDirectionCosines function used to specify the orientation of the slice. The direction cosines give the x, y, and z axes for the output volume
        :param center_point: tuple. Axes center for the reslicing operation in RAS format.
        :param target_size: tuple. Size of the output image.
        :param output_spacing: tuple. Spacing of the output image.
        :return: resliced vtk image in 3D
        """
        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(image)
        reslice.SetResliceAxesDirectionCosines(
            x_axis[0],
            x_axis[1],
            x_axis[2],
            y_axis[0],
            y_axis[1],
            y_axis[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
        )
        reslice.SetResliceAxesOrigin(center_point)
        reslice.SetOutputDimensionality(3)

        reslice.SetInterpolationMode(vtk.VTK_RESLICE_CUBIC)
        reslice.SetOutputSpacing(output_spacing)
        reslice.SetOutputExtent(
            0, target_size[0] - 1, 0, target_size[1] - 1, 0, target_size[2] - 1
        )
        reslice.SetOutputOrigin(
            -(target_size[0] * 0.5 - 0.5) * output_spacing[0],
            -(target_size[1] * 0.5 - 0.5) * output_spacing[1],
            -(target_size[2] * 0.5 - 0.5) * output_spacing[2],
        )
        reslice.Update()
        return reslice.GetOutput().GetPointData().GetScalars()

    @classmethod
    def reslice_2D_image_vtk(
        cls, image, x_axis, y_axis, z_axis, center_point, target_size, output_spacing
    ):
        """
        2D image reslicing using vtk.
        :param image: VTK image
        :param x_axis: tuple. X direction for vtk SetResliceAxesDirectionCosines function used to specify the orientation of the slice. The direction cosines give the x, y, and z axes for the output volume
        :param y_axis: tuple. Y direction for vtk SetResliceAxesDirectionCosines function used to specify the orientation of the slice. The direction cosines give the x, y, and z axes for the output volume
        :param z_axis: tuple. Z direction for vtk SetResliceAxesDirectionCosines function used to specify the orientation of the slice. The direction cosines give the x, y, and z axes for the output volume
        :param center_point: tuple. Axes center for the reslicing operation in RAS format.
        :param target_size: tuple. Size of the output image.
        :param output_spacing: tuple. Spacing of the output image in x, y , z.
        :return: resliced vtk image in 2D
        """
        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(image)
        reslice.SetResliceAxesDirectionCosines(
            x_axis[0],
            x_axis[1],
            x_axis[2],
            y_axis[0],
            y_axis[1],
            y_axis[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
        )
        reslice.SetResliceAxesOrigin(center_point)
        reslice.SetOutputDimensionality(2)

        reslice.SetInterpolationMode(vtk.VTK_RESLICE_CUBIC)
        reslice.SetOutputSpacing(output_spacing)
        reslice.SetOutputExtent(0, target_size[0] - 1, 0, target_size[1] - 1, 0, 1)
        reslice.SetOutputOrigin(
            -(target_size[0] * 0.5 - 0.5) * output_spacing[0],
            -(target_size[1] * 0.5 - 0.5) * output_spacing[1],
            0,
        )
        reslice.Update()
        return reslice.GetOutput().GetPointData().GetScalars()

    @classmethod
    def similarity_3D_transform_with_coords(
        cls,
        img,
        coords,
        output_size,
        translation,
        scale,
        interpolator=sitk.sitkBSpline,
        default_pixel_value=0.0,
    ):
        """
        Apply a 3D similarity transform to an image and use the same transformation for a list of coordinates
        (rotation not implemented at the moment)
        :param img: simpleITK image
        :param coords: numpy array of coordinates (Nx3) or None
        :param output_size:
        :param scale:
        :param translation:
        :return: tuple with sitkImage, transformed_coords
        """
        reference_image = sitk.Image(output_size, img.GetPixelIDValue())
        output_size_arr = np.array(output_size)
        reference_image.SetOrigin(img.GetOrigin())
        reference_image.SetDirection(img.GetDirection())
        spacing = (
            np.array(img.GetSize()) * np.array(img.GetSpacing())
        ) / output_size_arr
        reference_image.SetSpacing(spacing)

        # Create the transformation
        tr = sitk.Similarity3DTransform()
        if translation is not None:
            tr.SetTranslation(translation)
        if scale is not None:
            tr.SetScale(scale)

        # Apply the transformation to the image
        img2 = sitk.Resample(
            img, reference_image, tr, interpolator, default_pixel_value
        )

        if coords is not None:
            # Apply the transformation to the coordinates
            transformed_coords = np.zeros_like(coords)
            for i in range(coords.shape[0]):
                coords_ph = img.TransformContinuousIndexToPhysicalPoint(coords[i])
                coords_ph = tr.GetInverse().TransformPoint(coords_ph)
                transformed_coords[i] = np.array(
                    img2.TransformPhysicalPointToContinuousIndex(coords_ph)
                )
        else:
            transformed_coords = None

        return img2, transformed_coords

    @classmethod
    def scale_images(cls, img, output_size, return_scale_factors=False):
        """
        Scale an array that represents one or more images into a shape
        :param img: numpy array. It may contain one or multiple images
        :param output_size: tuple of int. Shape expected (including possibly the number of images)
        :param return_scale_factors: bool. If true, the result will be a tuple whose second values are the factors that
                                     were needed to downsample the images
        :return: numpy array rescaled or tuple with (array, factors)
        """
        img_size = np.array(img.shape)
        scale_factors = None
        if not np.array_equal(output_size, img_size):
            # The shape is the volume is different than the one expected by the network. We need to resize
            scale_factors = output_size / img_size
            # Reduce the volume to fit in the desired size
            img = scipy_interpolation.zoom(img, scale_factors)
        if return_scale_factors:
            return img, scale_factors
        return img

    @classmethod
    def standardization(cls, image_array, mean_value=-600, std_value=1.0, out=None):
        """
        Standarize an image substracting mean and dividing by variance
        :param image_array: image array
        :param mean_value: float. Image mean value. If None, ignore
        :param std_value: float. Image standard deviation value. If None, ignore
        :return: New numpy array unless 'out' parameter is used. If so, reference to that array
        """
        if out is None:
            # Build a new array (copy)
            image = image_array.astype(np.float32)
        else:
            # We will return a reference to out parameter
            image = out
            if id(out) != id(image_array):
                # The input and output arrays are different.
                # First, copy the source values, as we will apply the operations to image object
                image[:] = image_array[:]

        assert (
            image.dtype == np.float32
        ), "The out array must contain float32 elements, because the transformation will be performed in place"

        if mean_value is None:
            mean_value = image.mean()
        if std_value is None:
            std_value = image.std()
            if std_value <= 0.0001:
                std_value = 1.0

        # Standardize image
        image -= mean_value
        image /= std_value

        return image

    @classmethod
    def normalize_CT_image_intensity(
        cls,
        image_array,
        min_value=-300,
        max_value=700,
        min_output=0.0,
        max_output=1.0,
        out=None,
    ):
        """
        Threshold and adjust contrast range in a CT image.
        :param image_array: int numpy array (CT or partial CT image)
        :param min_value: int. Min threshold (everything below that value will be thresholded). If None, ignore
        :param max_value: int. Max threshold (everything below that value will be thresholded). If None, ignore
        :param min_output: float. Min out value
        :param max_output: float. Max out value
        :param out: numpy array. Array that will be used as an output
        :return: New numpy array unless 'out' parameter is used. If so, reference to that array
        """
        clip = min_value is not None or max_value is not None
        if min_value is None:
            min_value = np.min(image_array)
        if max_value is None:
            max_value = np.max(image_array)

        if out is None:
            # Build a new array (copy)
            image = image_array.astype(np.float32)
        else:
            # We will return a reference to out parameter
            image = out
            if id(out) != id(image_array):
                # The input and output arrays are different.
                # First, copy the source values, as we will apply the operations to image object
                image[:] = image_array[:]

        assert (
            image.dtype == np.float32
        ), "The out array must contain float32 elements, because the transformation will be performed in place"

        if clip:
            np.clip(image, min_value, max_value, image)

        # Change of range
        image -= min_value
        image /= max_value - min_value
        image *= max_output - min_output
        image += min_output

        return image

    @classmethod
    def elastic_transform(cls, image, alpha, sigma, fill_mode="constant", cval=0.0):
        """
        Elastic deformation of images as described in  http://doi.ieeecomputersociety.org/10.1109/ICDAR.2003.1227801
        :param image: numpy array
        :param alpha: float
        :param sigma: float
        :param fill_mode: fill mode for gaussian filer. Default: constant value (cval)
        :param cval: float
        :return: numpy array. Image transformed
        """
        random_state = np.random.RandomState(None)
        shape = image.shape
        dx = (
            gaussian_filter(
                (random_state.rand(*shape) * 2 - 1), sigma, mode=fill_mode, cval=cval
            )
            * alpha
        )
        dy = (
            gaussian_filter(
                (random_state.rand(*shape) * 2 - 1), sigma, mode=fill_mode, cval=cval
            )
            * alpha
        )
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        distorted_image = map_coordinates(image, indices, order=1).reshape(shape)
        return distorted_image

    @classmethod
    def elastic_deformation_2D(
        cls, image, grid_width=2, grid_height=2, magnitude=4, resampling="bicubic"
    ):
        """
        Distorts a 2D image according to the parameters and returns the newly distorted image. Class taken from Augmentor methods
        :param image:
        :param grid_width: int. Grid width
        :param grid_height: int. Grid height
        :param magnitude: int. Magnitude
        :param resampling: str. Resampling filter. Options: nearest (use nearest neighbour) |
        bilinear (linear interpolation in a 2x2 environment) | bicubic (cubic spline interpolation in a 4x4 environment)
        """
        image = Image.fromarray(image.transpose())
        w, h = image.size

        horizontal_tiles = grid_width
        vertical_tiles = grid_height

        width_of_square = int(math.floor(w / float(horizontal_tiles)))
        height_of_square = int(math.floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (
                    horizontal_tiles - 1
                ):
                    dimensions.append(
                        [
                            horizontal_tile * width_of_square,
                            vertical_tile * height_of_square,
                            width_of_last_square + (horizontal_tile * width_of_square),
                            height_of_last_square + (height_of_square * vertical_tile),
                        ]
                    )
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append(
                        [
                            horizontal_tile * width_of_square,
                            vertical_tile * height_of_square,
                            width_of_square + (horizontal_tile * width_of_square),
                            height_of_last_square + (height_of_square * vertical_tile),
                        ]
                    )
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append(
                        [
                            horizontal_tile * width_of_square,
                            vertical_tile * height_of_square,
                            width_of_last_square + (horizontal_tile * width_of_square),
                            height_of_square + (height_of_square * vertical_tile),
                        ]
                    )
                else:
                    dimensions.append(
                        [
                            horizontal_tile * width_of_square,
                            vertical_tile * height_of_square,
                            width_of_square + (horizontal_tile * width_of_square),
                            height_of_square + (height_of_square * vertical_tile),
                        ]
                    )

        # For loop that generates polygons could be rewritten, but maybe harder to read?
        # polygons = [x1,y1, x1,y2, x2,y2, x2,y1 for x1,y1, x2,y2 in dimensions]
        # last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range(
            (horizontal_tiles * vertical_tiles) - horizontal_tiles,
            horizontal_tiles * vertical_tiles,
        )

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append(
                    [i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles]
                )

        for a, b, c, d in polygon_indices:
            dx = np.random.randint(-magnitude, magnitude)
            dy = np.random.randint(-magnitude, magnitude)

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1, x2, y2, x3 + dx, y3 + dy, x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1, x2 + dx, y2 + dy, x3, y3, x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1, x2, y2, x3, y3, x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy, x2, y2, x3, y3, x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        if resampling == "bilinear":
            resampling_filter = Image.BILINEAR
        elif resampling == "nearest":
            resampling_filter = Image.NEAREST
        else:
            resampling_filter = Image.BICUBIC

        return np.asarray(
            image.transform(
                image.size, Image.MESH, generated_mesh, resample=resampling_filter
            )
        ).transpose()

    @classmethod
    def perspective_skew_2D_transform(
        cls, image, skew_amount, skew_type="random", resampling="bicubic"
    ):
        """
        Apply perspective skewing on images. Class taken from Augmentor methods
        :param image:
        :param skew_type: str. Skew type. Options: random | tilt (will randomly skew either left, right, up, or down.) |
        tilt_top_buttton (skew up or down) | tilt_left_right (skew left or right) |
        corner (will randomly skew one **corner** of the image either along the x-axis or y-axis.
        This means in one of 8 different directions, randomly.
        :param skew_amount: int. The degree to which the image is skewed
        :param resampling: str. Resampling filter. Options: nearest (use nearest neighbour) |
        bilinear (linear interpolation in a 2x2 environment) | bicubic (cubic spline interpolation in a 4x4 environment)
        """
        image = Image.fromarray(image.transpose())

        w, h = image.size

        x1 = 0
        x2 = h
        y1 = 0
        y2 = w

        original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

        if skew_type == "random":
            skew = np.random.choice(
                ["tilt", "tilt_left_right", "tilt_top_buttton", "corner"]
            )
        else:
            skew = skew_type

        # We have two choices now: we tilt in one of four directions
        # or we skew a corner.

        if skew == "tilt" or skew == "tilt_left_right" or skew == "tilt_top_buttton":
            if skew == "tilt":
                skew_direction = np.random.randint(0, 3)
            elif skew == "tilt_left_right":
                skew_direction = np.random.randint(0, 1)
            elif skew == "tilt_top_buttton":
                skew_direction = np.random.randint(2, 3)

            if skew_direction == 0:
                # Left Tilt
                new_plane = [
                    (y1, x1 - skew_amount),  # Top Left
                    (y2, x1),  # Top Right
                    (y2, x2),  # Bottom Right
                    (y1, x2 + skew_amount),
                ]  # Bottom Left
            elif skew_direction == 1:
                # Right Tilt
                new_plane = [
                    (y1, x1),  # Top Left
                    (y2, x1 - skew_amount),  # Top Right
                    (y2, x2 + skew_amount),  # Bottom Right
                    (y1, x2),
                ]  # Bottom Left
            elif skew_direction == 2:
                # Forward Tilt
                new_plane = [
                    (y1 - skew_amount, x1),  # Top Left
                    (y2 + skew_amount, x1),  # Top Right
                    (y2, x2),  # Bottom Right
                    (y1, x2),
                ]  # Bottom Left
            elif skew_direction == 3:
                # Backward Tilt
                new_plane = [
                    (y1, x1),  # Top Left
                    (y2, x1),  # Top Right
                    (y2 + skew_amount, x2),  # Bottom Right
                    (y1 - skew_amount, x2),
                ]  # Bottom Left

        if skew == "corner":
            skew_direction = np.random.randint(0, 7)

            if skew_direction == 0:
                # Skew possibility 0
                new_plane = [(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)]
            elif skew_direction == 1:
                # Skew possibility 1
                new_plane = [(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)]
            elif skew_direction == 2:
                # Skew possibility 2
                new_plane = [(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)]
            elif skew_direction == 3:
                # Skew possibility 3
                new_plane = [(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)]
            elif skew_direction == 4:
                # Skew possibility 4
                new_plane = [(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)]
            elif skew_direction == 5:
                # Skew possibility 5
                new_plane = [(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)]
            elif skew_direction == 6:
                # Skew possibility 6
                new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)]
            elif skew_direction == 7:
                # Skew possibility 7
                new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)]

        # To calculate the coefficients required by PIL for the perspective skew,
        # see the following Stack Overflow discussion: https://goo.gl/sSgJdj
        matrix = []

        for p1, p2 in zip(new_plane, original_plane):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(original_plane).reshape(8)

        perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
        perspective_skew_coefficients_matrix = np.array(
            perspective_skew_coefficients_matrix
        ).reshape(8)

        if resampling == "bilinear":
            resampling_filter = Image.BILINEAR
        elif resampling == "nearest":
            resampling_filter = Image.NEAREST
        else:
            resampling_filter = Image.BICUBIC

        return np.asarray(
            image.transform(
                image.size,
                Image.PERSPECTIVE,
                perspective_skew_coefficients_matrix,
                resample=resampling_filter,
            )
        ).transpose()
