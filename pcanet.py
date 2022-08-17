# [the original paper](https://arxiv.org/abs/1404.3606)

import itertools

from chainer.cuda import to_gpu, to_cpu
from chainer.functions import convolution_2d

import numpy as np
from sklearn.decomposition import IncrementalPCA

# from utils import gpu_enabled
import numpy as np
import pickle

try:
    import importlib
except ImportError:
    import imp as importlib

from chainer.datasets import get_mnist, get_cifar10
from chainer.cuda import get_device


GPU_ENABLED = False


def set_device(device_id):
    """
    Set the device (CPU or GPU) to be used.
    if device_id >= 0 the corresponding GPU is used, otherwise CPU is used.
    """
    if device_id < 0:
        # Use CPU
        return

    try:
        from cupy.cuda import Device
        from cupy.cuda.runtime import CUDARuntimeError
    except ImportError:
        print("Failed to import CuPy. Use CPU instead.")
        return

    try:
        Device(device_id).use()
    except CUDARuntimeError as e:
        print(e)
        return

    print("Device {} is in use".format(device_id))

    global GPU_ENABLED
    GPU_ENABLED = True

    # Reload the module to reflect the GPU status
    import pcanet
    importlib.reload(pcanet)


def gpu_enabled():
    return GPU_ENABLED


def reshape_dataset(train, test):
    def channels_last(X):
        X = np.swapaxes(X, 1, 2)
        X = np.swapaxes(X, 2, 3)
        return X

    X_train, y_train = train._datasets[0], train._datasets[1]
    X_test, y_test = test._datasets[0], test._datasets[1]
    X_train, X_test = channels_last(X_train), channels_last(X_test)
    return ((X_train, y_train), (X_test, y_test))


def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model


def pick(train_set, test_set, n_train, n_test):
    images_train, y_train = train_set
    images_test, y_test = test_set
    train_set = images_train[:n_train], y_train[:n_train]
    test_set = images_test[:n_test], y_test[:n_test]
    return train_set, test_set


def load_cifar():
    train, test = get_cifar10(ndim=3)
    return reshape_dataset(train, test)


def load_mnist():
    train, test = get_mnist(ndim=3)
    return reshape_dataset(train, test)


def concatenate_dicts(*dicts):
    """Concatenate multiple directories into one"""
    merged = []
    for d in dicts:
        merged += list(d.items())
    return dict(merged)


if gpu_enabled():
    try:
        import cupy as xp
    except ImportError:
        import numpy as xp
else:
    import numpy as xp


def steps(image_shape, filter_shape, step_shape):
    """
    Generates feature map coordinates that filters visit

    Parameters
    ----------
    image_shape: tuple of ints
        Image height / width
    filter_shape: tuple of ints
        Filter height / width
    step_shape: tuple of ints
        Step height / width

    Returns
    -------
    ys: Map coordinates along y axis
    xs: Map coordinates along x axis
    """
    h, w = image_shape
    fh, fw = filter_shape
    sh, sw = step_shape

    ys = range(0, h-fh+1, sh)
    xs = range(0, w-fw+1, sw)
    return ys, xs


def components_to_filters(components, n_channels, filter_shape):
    """
    | In PCANet, components of PCA are used as filter weights.
    | This function reshapes PCA components so that
      it can be used as networks filters
    """
    n_filters = components.shape[0]
    return components.reshape(n_filters, n_channels, *filter_shape)


def output_shape(ys, xs):
    return len(ys), len(xs)


class Patches(object):
    def __init__(self, image, filter_shape, step_shape):
        assert(image.ndim == 2)

        # should be either numpy.ndarray or cupy.ndarray
        self.ndarray = type(image)
        self.image = image
        self.filter_shape = filter_shape

        self.ys, self.xs = steps(image.shape[0:2], filter_shape, step_shape)

    @property
    def patches(self):
        """
        Return image patches of shape
        (n_patches, filter_height, filter_width)
        """
        fh, fw = self.filter_shape
        it = list(itertools.product(self.ys, self.xs))
        patches = self.ndarray((len(it), fh, fw), dtype=self.image.dtype)
        for i, (y, x) in enumerate(it):
            patches[i, :, :] = self.image[y:y+fh, x:x+fw]
        return patches

    @property
    def output_shape(self):
        return output_shape(self.ys, self.xs)


def atleast_4d(images):
    """Regard gray-scale images as 1-channel images"""
    assert(np.ndim(images) == 3)
    n, h, w = images.shape
    return images.reshape(n, h, w, 1)


def to_channels_first(images):
    """
    Change image channel order from
    :code:`(n_images, y, x, n_channels)` to
    :code:`(n_images, n_channels, y, x)`
    """
    # images.shape == (n_images, y, x, n_channels)
    images = np.swapaxes(images, 1, 3)
    images = np.swapaxes(images, 2, 3)
    # images.shape == (n_images, n_channels, y, x)
    return images


def image_to_patch_vectors(image, filter_shape, step_shape):
    """
    Parameters
    ----------
    image: np.ndarray
        Image to extract patch vectors
    filter_shape: tuple of ints
        The shape of a filter
    step_shape: tuple of ints
        Step height/width of a filter

    Returns
    -------
    X: np.ndarray
        A set of normalized and flattened patches
    """

    X = Patches(image, filter_shape, step_shape).patches
    # X.shape == (n_patches, filter_height, filter_width)

    X = X.reshape(X.shape[0], -1)  # flatten each patch
    X = X - X.mean(axis=1, keepdims=True)  # Remove mean from each patch.
    return X  # \overline{X}_i in the original paper


def binarize(X):
    """
    Binarize each element of :code:`X`

    .. code::

        X = [1 if X[i] > 0 else 0 for i in range(len(X))]
    """
    X[X > 0] = 1
    X[X <= 0] = 0
    return X


def binary_to_decimal(X):
    """
    | This function takes :code:`X` of shape (n_images, L2, y, x) as an argument.
    | Supporse that :code:`X[k]` (0 <= k < n_images) can be represented as

    .. code-block:: none

        X[k] = [map_k[0], map_k[1], ..., map_k[L2-1]]

    where the shape of each map_k is (y, x).

    Then we calculate

    .. code-block:: none

        a[0] * map_k[0] + a[1] * map_k[1] + ... + a[L2-1] * map_k[L2-1]

    for each :code:`X[k]`, where :math:`a = [2^{L2-1}, 2^{L2-2}, ..., 2^{0}]`

    Therefore, the output shape must be (n_images, y, x)

    Parameters
    ----------
    X: xp.ndarray
        Feature maps
    """
    a = xp.arange(X.shape[1])[::-1]
    a = xp.power(2, a)
    return xp.tensordot(X, a, axes=([1], [0]))


def to_tuple_if_int(value):
    """
    If int is given, duplicate it and return as a 2 element tuple.
    """
    if isinstance(value, int):
        return (value, value)
    return value


class PCANet(object):
    def __init__(self, image_shape,
                 filter_shape_l1, step_shape_l1, n_l1_output,
                 filter_shape_l2, step_shape_l2, n_l2_output,
                 filter_shape_pooling, step_shape_pooling):
        """
        Parameters
        ----------
        image_shape: int or sequence of ints
            Input image shape.
        filter_shape_l1: int or sequence of ints
            The shape of the kernel in the first convolution layer.
            If the value is int, a filter of the square shape is applied.
            If you want to apply a filter of a different aspect ratio, just
            pass a tuple of shape (height, width).
        step_shape_l1: int or sequence of ints
            The shape of kernel step in the first convolution layer.
            If the value is int, a step of the square shape is applied.
            If you want to apply a step of a different aspect ratio, just
            pass a tuple of shape (height, width).
        n_l1_output:
            L1 in the original paper. The number of outputs obtained
            from a set of input images.
        filter_shape_l2: int or sequence of ints
            The shape of the kernel in the second convolution layer.
            If the value is int, a filter of the square shape is applied.
            If you want to apply a filter of a different aspect ratio, just
            pass a tuple of shape (height, width).
        step_shape_l2: int or sequence of ints
            The shape of kernel step in the second convolution layer.
            If the value is int, a step of the square shape is applied.
            If you want to apply a step of a different aspect ratio, just
            pass a tuple of shape (height, width).
        n_l2_output:
            L2 in the original paper. The number of outputs obtained
            from each L1 output.
        filter_shape_pooling: int or sequence of ints
            The shape of the filter in the pooling layer.
        step_shape_pooling: int or sequence of ints
            The shape of the filter step in the pooling layer.
        """

        self.image_shape = to_tuple_if_int(image_shape)

        self.filter_shape_l1 = to_tuple_if_int(filter_shape_l1)
        self.step_shape_l1 = to_tuple_if_int(step_shape_l1)
        self.n_l1_output = n_l1_output

        self.filter_shape_l2 = to_tuple_if_int(filter_shape_l2)
        self.step_shape_l2 = to_tuple_if_int(step_shape_l2)
        self.n_l2_output = n_l2_output

        self.filter_shape_pooling = to_tuple_if_int(filter_shape_pooling)
        self.step_shape_pooling = to_tuple_if_int(step_shape_pooling)
        self.n_bins = None  # TODO make n_bins specifiable

        self.pca_l1 = IncrementalPCA(n_l1_output)
        self.pca_l2 = IncrementalPCA(n_l2_output)

    def histogram(self, binary_images):
        """
        Separate a given image into blocks and calculate a histogram
        in each block.

        Supporse data in a block is in range [0, 3] and the acutual
        values are

        ::

            [0 0 1]
            [2 2 2]
            [2 3 3]

        | If default bins ``[-0.5 0.5 1.5 2.5 3.5]`` applied,
          the histogram will be ``[2 1 4 2]``.
        | If ``n_bins`` is specified, the range of data divided equally.

        | For example, if the data is in range ``[0, 3]`` and ``n_bins = 2``,
        | bins will be ``[-0.5 1.5 3.5]`` and the histogram will be ``[3 6]``.
        """

        k = pow(2, self.n_l2_output)
        if self.n_bins is None:
            self.n_bins = k + 1
        bins = xp.linspace(-0.5, k - 0.5, self.n_bins)

        def bhist(image):
            # calculate Bhist(T) in the original paper
            ps = Patches(
                image,
                self.filter_shape_pooling,
                self.step_shape_pooling).patches

            H = [xp.histogram(p.flatten(), bins)[0] for p in ps]
            return xp.concatenate(H)
        return xp.vstack([bhist(image) for image in binary_images])

    def process_input(self, images):
        assert(np.ndim(images) >= 3)
        assert(images.shape[1:3] == self.image_shape)
        if np.ndim(images) == 3:
            # forcibly convert to multi-channel images
            images = atleast_4d(images)
        images = to_channels_first(images)
        return images

    def fit(self, images):
        """
        Train PCANet

        Parameters
        ----------
        images: np.ndarray
            | Color / grayscale images of shape
            | (n_images, height, width, n_channels) or
            | (n_images, height, width)
        """
        images = self.process_input(images)
        # images.shape == (n_images, n_channels, y, x)

        for image in images:
            X = []
            for channel in image:
                patches = image_to_patch_vectors(
                    channel,
                    self.filter_shape_l1,
                    self.step_shape_l1
                )
                X.append(patches)
            patches = np.hstack(X)
            # patches.shape = (n_patches, n_patches * vector length)
            self.pca_l1.partial_fit(patches)

        filters_l1 = components_to_filters(
            self.pca_l1.components_,
            n_channels=images.shape[1],
            filter_shape=self.filter_shape_l1,
        )

        if gpu_enabled():
            images = to_gpu(images)
            filters_l1 = to_gpu(filters_l1)

        images = convolution_2d(
            images,
            filters_l1,
            stride=self.step_shape_l1
        ).data

        if gpu_enabled():
            images = to_cpu(images)
            filters_l1 = to_cpu(filters_l1)

        # images.shape == (n_images, L1, y, x)
        images = images.reshape(-1, *images.shape[2:4])

        for image in images:
            patches = image_to_patch_vectors(
                image,
                self.filter_shape_l2,
                self.step_shape_l2
            )
            self.pca_l2.partial_fit(patches)
        return self

    def transform(self, images):
        """
        Parameters
        ----------
        images: np.ndarray
            | Color / grayscale images of shape
            | (n_images, height, width, n_channels) or
            | (n_images, height, width)

        Returns
        -------
        X: np.ndarray
            A set of feature vectors of shape (n_images, n_features)
            where :code:`n_features` is determined by the hyperparameters
        """
        images = self.process_input(images)
        # images.shape == (n_images, n_channels, y, x)

        filters_l1 = components_to_filters(
            self.pca_l1.components_,
            n_channels=images.shape[1],
            filter_shape=self.filter_shape_l1,
        )

        filters_l2 = components_to_filters(
            self.pca_l2.components_,
            n_channels=1,
            filter_shape=self.filter_shape_l2
        )

        if gpu_enabled():
            images = to_gpu(images)
            filters_l1 = to_gpu(filters_l1)
            filters_l2 = to_gpu(filters_l2)

        images = convolution_2d(
            images,
            filters_l1,
            stride=self.step_shape_l1
        ).data

        images = xp.swapaxes(images, 0, 1)

        # L1.shape == (L1, n_images, y, x)
        # iterate over each L1 output

        X = []
        for maps in images:
            n_images, h, w = maps.shape
            maps = convolution_2d(
                maps.reshape(n_images, 1, h, w),  # 1 channel images
                filters_l2,
                stride=self.step_shape_l2
            ).data

            # maps.shape == (n_images, L2, y, x) right here
            maps = binarize(maps)
            maps = binary_to_decimal(maps)
            # maps.shape == (n_images, y, x)
            x = self.histogram(maps)

            # x is a set of feature vectors.
            # The shape of x is (n_images, vector length)
            X.append(x)

        # concatenate over L1
        X = xp.hstack(X)

        if gpu_enabled():
            X = to_cpu(X)

        X = X.astype(np.float64)

        # The shape of X is (n_images, L1 * vector length)
        return X

    def validate_structure(self):
        """
        Check that the filter visits all pixels of input images without
        dropping any information.

        Raises
        ------
        ValueError:
            if the network structure does not satisfy the above constraint.
        """
        def is_valid_(input_shape, filter_shape, step_shape):
            ys, xs = steps(input_shape, filter_shape, step_shape)
            fh, fw = filter_shape
            h, w = input_shape
            if ys[-1]+fh != h or xs[-1]+fw != w:
                raise ValueError("Invalid network structure.")
            return output_shape(ys, xs)

        output_shape_l1 = is_valid_(self.image_shape,
                                    self.filter_shape_l1,
                                    self.step_shape_l1)
        output_shape_l2 = is_valid_(output_shape_l1,
                                    self.filter_shape_l2,
                                    self.step_shape_l2)
        is_valid_(
            output_shape_l2,
            self.filter_shape_pooling,
            self.filter_shape_pooling
        )
