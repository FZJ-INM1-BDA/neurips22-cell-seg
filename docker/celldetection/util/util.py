import numpy as np
import inspect
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Union, List, Tuple, Any, Dict as TDict, Iterator, Type
from torch import Tensor
from torch.hub import load_state_dict_from_url
import hashlib
import json
from tqdm import tqdm
from os.path import join
from os import makedirs
import pynvml as nv
import h5py

__all__ = ['Dict', 'lookup_nn', 'reduce_loss_dict', 'tensor_to', 'to_device', 'asnumpy', 'fetch_model',
           'random_code_name', 'dict_hash', 'fetch_image', 'random_seed', 'tweak_module_', 'add_to_loss_dict',
           'random_code_name_dir', 'get_device', 'num_params', 'count_submodules', 'train_epoch', 'Bytes', 'Percent',
           'GpuStats', 'trainable_params', 'frozen_params', 'Tiling', 'load_image',
           'iter_submodules', 'replace_module_', 'wrap_module_', 'spectral_norm_', 'to_h5', 'to_tiff',
           'exponential_moving_average_', 'from_json', 'to_json', 'weight_norm_', 'inject_extra_repr_', 'NormProxy']


class Dict(dict):
    __getattr__ = dict.__getitem__  # alternative: dict.get if KeyError is not desired
    __delattr__ = dict.__delitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        """Dictionary.

        Just a ``dict`` that treats values like attributes.

        Examples:
            >>> import celldetection as cd
            >>> d = cd.Dict(my_value=42)
            >>> d.my_value
            42
            >>> d.my_value += 1
            >>> d.my_value
            43

        Args:
            **kwargs:
        """
        super().__init__(kwargs)


def lookup_nn(item: str, *a, src=None, call=True, inplace=True, **kw):
    """

    Examples:
        >>> lookup_nn('batchnorm2d', 32)
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> lookup_nn(torch.nn.BatchNorm2d, 32)
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> lookup_nn('batchnorm2d', num_features=32)
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> lookup_nn('tanh')
            Tanh()
        >>> lookup_nn('tanh', call=False)
            torch.nn.modules.activation.Tanh
        >>> lookup_nn('relu')
            ReLU(inplace=True)
        >>> lookup_nn('relu', inplace=False)
            ReLU()

    Args:
        item: Lookup item. None is equivalent to `identity`.
        *a: Arguments passed to item if called.
        src: Lookup source.
        call: Whether to call item.
        inplace: Default setting for items that take an `inplace` argument when called.
            As default is True, `lookup_nn('relu')` returns a ReLu instance with `inplace=True`.
        **kw: Keyword arguments passed to item when it is called.

    Returns:
        Looked up item.
    """
    src = src or nn
    if item is None:
        v = nn.Identity
    elif isinstance(item, str):
        l_item = item.lower()
        v = next((getattr(src, i) for i in dir(src) if i.lower() == l_item))
    elif isinstance(item, nn.Module):
        return item
    else:
        v = item
    if call:
        kwargs = {'inplace': inplace} if 'inplace' in inspect.getfullargspec(v).args else {}
        kwargs.update(kw)
        v = v(*a, **kwargs)
    return v


class NormProxy:
    def __init__(self, norm, **kwargs):
        """Norm Proxy.

        Examples:
            >>> GroupNorm = NormProxy('groupnorm', num_groups=32)
            ... GroupNorm(3)
            GroupNorm(32, 3, eps=1e-05, affine=True)
            >>> GroupNorm = NormProxy(nn.GroupNorm, num_groups=32)
            ... GroupNorm(3)
            GroupNorm(32, 3, eps=1e-05, affine=True)
            >>> BatchNorm2d = NormProxy('batchnorm2d', momentum=.2)
            ... BatchNorm2d(3)
            BatchNorm2d(3, eps=1e-05, momentum=0.2, affine=True, track_running_stats=True)
            >>> BatchNorm2d = NormProxy(nn.BatchNorm2d, momentum=.2)
            ... BatchNorm2d(3)
            BatchNorm2d(3, eps=1e-05, momentum=0.2, affine=True, track_running_stats=True)

        Args:
            norm: Norm class or name.
            **kwargs: Keyword arguments.
        """
        self.norm = norm
        self.kwargs = kwargs

    def __call__(self, num_channels):
        Norm = lookup_nn(self.norm, call=False)
        kwargs = dict(self.kwargs)
        args = inspect.getfullargspec(Norm).args
        if 'num_features' in args:
            kwargs['num_features'] = num_channels
        elif 'num_channels' in args:
            kwargs['num_channels'] = num_channels
        return Norm(**kwargs)

    def __repr__(self):
        return f'NormProxy({self.norm}, kwargs={self.kwargs})'

    __str__ = __repr__


def reduce_loss_dict(losses: dict, divisor):
    return sum((i for i in losses.values() if i is not None)) / divisor


def add_to_loss_dict(d: dict, key: str, loss: torch.Tensor, weight=None):
    dk = d[key]
    torch.nan_to_num_(loss, 0., 0., 0.)
    if weight is not None:
        loss = loss * weight
    d[key] = loss if dk is None else dk + loss


def tensor_to(inputs: Union[list, tuple, dict, Tensor], *args, **kwargs):
    """Tensor to device/dtype/other.

    Recursively calls ``tensor.to(*args, **kwargs)`` for all ``Tensors`` in ``inputs``.

    Notes:
        - Works recursively.
        - Non-Tensor items are not altered.

    Args:
        inputs: Tensor, list, tuple or dict. Non-Tensor objects are ignored. Tensors are substituted by result of
            ``tensor.to(*args, **kwargs)`` call.
        *args: Arguments. See docstring of ``torch.Tensor.to``.
        **kwargs: Keyword arguments. See docstring of ``torch.Tensor.to``.

    Returns:
        Inputs with Tensors replaced by ``tensor.to(*args, **kwargs)``.
    """
    if isinstance(inputs, Tensor):
        inputs = inputs.to(*args, **kwargs)
    elif isinstance(inputs, dict):
        inputs = {k: tensor_to(b, *args, **kwargs) for k, b in inputs.items()}
    elif isinstance(inputs, (list, tuple)):
        inputs = type(inputs)([tensor_to(b, *args, **kwargs) for b in inputs])
    return inputs


def to_device(batch: Union[list, tuple, dict, Tensor], device):
    """To device.

    Move Tensors to device.
    Input can be Tensor, tuple of Tensors, list of Tensors or a dictionary of Tensors.

    Notes:
        - Works recursively.
        - Non-Tensor items are not altered.

    Args:
        batch: Tensor, list, tuple or dict. Non-Tensor objects are ignored. Tensors are moved to ``device``.
        device: Device.

    Returns:
        Input with Tensors moved to device.
    """
    return tensor_to(batch, device)


def asnumpy(v):
    """As numpy.

    Converts all Tensors to numpy arrays.

    Notes:
        - Works recursively.
        - The following input items are not altered: Numpy array, int, float, bool, str

    Args:
        v: Tensor or list/tuple/dict of Tensors.

    Returns:
        Input with Tensors converted to numpy arrays.
    """
    if v is None:
        return v
    elif isinstance(v, torch.Tensor):
        if str(v.device) != 'cpu':
            v = v.cpu()
        return v.data.numpy()
    elif isinstance(v, (np.ndarray, int, float, bool, np.float, np.int, np.bool, str)):
        return v
    elif isinstance(v, (tuple, list)):
        return [asnumpy(val) for val in v]
    elif isinstance(v, dict):
        r = dict()
        for k, val in v.items():
            r[k] = asnumpy(val)
        return r
    else:
        raise ValueError(f'Type not supported: {type(v)}')


def fetch_model(name, map_location=None, **kwargs):
    """Fetch model from URL.

    Loads model or state dict from URL.

    Args:
        name: Model name hosted on `celldetection.org` or url. Urls must start with 'http'.
        map_location: A function, `torch.device`, string or a dict specifying how to remap storage locations.
        **kwargs: From the doc of `torch.models.utils.load_state_dict_from_url`.

    """
    url = name if name.startswith('http') else f'https://celldetection.org/torch/models/{name}.pt'
    m = load_state_dict_from_url(url, map_location=map_location, **kwargs)
    if isinstance(m, dict) and 'cd.models' in m.keys():
        from .. import models
        conf = m['cd.models']
        m = getattr(models, conf['model'])(*conf['a'], **conf['kw'])
        m.load_state_dict(conf['state_dict'])
    return m


def random_code_name(chars=4) -> str:
    """Random code name.

    Generates random code names that are somewhat pronounceable and memorable.

    Examples:
        >>> import celldetection as cd
        >>> cd.random_code_name()
        kolo
        >>> cd.random_code_name(6)
        lotexo

    Args:
        chars: Number of characters.

    Returns:
        String.
    """
    a, b = [i for i in 'aeiou'], [i for i in 'tskyrhzjgqmxlvnfcpwbd']
    return ''.join([np.random.choice(b if j % 2 == 0 else a) for j in range(chars)])


def random_code_name_dir(directory='./out', chars=6):
    """Random code name directory.

    Creates random code name and creates a subdirectory with said name under `directory`.
    Code names that are already taken (subdirectory already exists) are not reused.

    Args:
        directory: Root directory.
        chars: Number of characters for the code name.

    Returns:
        Tuple of code name and created directory.
    """
    try:
        code_name = random_code_name(chars=chars)
        out_dir = join(directory, code_name)
        makedirs(out_dir)
    except FileExistsError:
        return random_code_name_dir(directory)
    return code_name, out_dir


def dict_hash(dictionary: TDict[str, Any]) -> str:
    """MD5 hash of a dictionary.

    References:
        https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html

    Args:
        dictionary: A dictionary.

    Returns:
        Md5 hash of the dictionary as string.
    """
    dhash = hashlib.md5()
    dhash.update(json.dumps(dictionary, sort_keys=True).encode())
    return dhash.hexdigest()


def fetch_image(url, numpy=True):
    """Fetch image from URL.

    Download an image from URL and convert it to a numpy array or PIL Image.

    Args:
        url: URL
        numpy: Whether to convert PIL Image to numpy array.

    Returns:
        PIL Image or numpy array.
    """
    import requests
    from PIL import Image
    img = Image.open(requests.get(url, stream=True).raw)
    return np.asarray(img) if numpy else img


def load_image(name, method='imageio') -> np.ndarray:
    """Load image.

    Load image from URL or from filename via ``imageio`` or ``pytiff``.

    Args:
        name: URL (must start with ``http``) or filename.
        method: Method to use for filenames.

    Returns:
        Image.
    """
    if name.startswith('http'):
        img = fetch_image(name)
    elif method == 'imageio':
        from imageio import imread
        img = imread(name)
    elif method == 'pytiff':
        from pytiff import Tiff
        with Tiff(name, 'r') as t:
            img = t[:]
    else:
        raise ValueError(f'Could not load {name} with method {method}. Also note that URLs should start with "http".')
    return img


def random_seed(seed, backends=False, deterministic_torch=True):
    """Set random seed.

    Set random seed to ``random``, ``np.random``, ``torch.backends.cudnn`` and ``torch.manual_seed``.
    Also advise torch to use deterministic algorithms.

    References:
        https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        seed: Random seed.
        backends: Whether to also adapt backends. If set True cuDNN's benchmark feature is disabled. This
            causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
            Also the selected algorithm is set to run deterministically.
        deterministic_torch: Whether to set PyTorch operations to behave deterministically.

    """
    from torch import manual_seed
    from torch.backends import cudnn
    import random
    random.seed(seed)
    manual_seed(seed)
    np.random.seed(seed)
    if backends:
        cudnn.deterministic = True
        cudnn.benchmark = False
    if deterministic_torch and 'use_deterministic_algorithms' in dir(torch):
        torch.use_deterministic_algorithms(True)


def train_epoch(model, train_loader, device, optimizer, desc=None, scaler=None, scheduler=None, gpu_stats=False):
    """Basic train function.

    Notes:
        - Model should return dictionary: {'loss': Tensor[], ...}
        - Batch from `train_loader` should be a dictionary: {'inputs': Tensor[...], ...}
        - Model must be callable: `model(batch['inputs'], targets=batch)`

    Args:
        model: Model.
        train_loader: Data loader.
        device: Device.
        optimizer: Optimizer.
        desc: Description, appears in progress print.
        scaler: Gradient scaler. If set PyTorch's autocast feature is used.
        scheduler: Scheduler. Step called after epoch.
        gpu_stats: Whether to print GPU stats.
    """
    from torch.cuda.amp import autocast
    model.train()
    tq = tqdm(train_loader, desc=desc)
    gpu_st = None
    if gpu_stats:
        gpu_st = GpuStats()
    for batch_idx, batch in enumerate(tq):
        batch: dict = to_device(batch, device)
        optimizer.zero_grad()
        with autocast(scaler is not None):
            outputs: dict = model(batch['inputs'], targets=batch)
        loss = outputs['loss']
        info = [] if desc is None else [desc]
        if gpu_st is not None:
            info.append(str(gpu_st))
        losses = outputs.get('losses')
        if losses is not None and isinstance(losses, dict):
            info.append('losses(' + ', '.join(
                [(f'{k}: %g' % np.round(asnumpy(v), 3)) for k, v in losses.items() if v is not None]) + ')')
        info.append('loss %g' % np.round(asnumpy(loss), 3))
        tq.desc = ' - '.join(info)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    if scheduler is not None:
        scheduler.step()


def iter_submodules(module: nn.Module, class_or_tuple, recursive=True):
    for k, mod in module._modules.items():
        if isinstance(mod, class_or_tuple):
            yield module._modules, k, mod
        if isinstance(mod, nn.Module) and recursive:
            yield from iter_submodules(mod, class_or_tuple, recursive=recursive)


def tweak_module_(module: nn.Module, class_or_tuple, must_exist=True, recursive=True, **kwargs):
    """Tweak module.

    Set attributes for all modules that are instances of given `class_or_tuple`.

    Examples:
        >>> import celldetection as cd, torch.nn as nn
        >>> model = cd.models.ResNet18(in_channels=3)
        >>> cd.tweak_module_(model, nn.BatchNorm2d, momentum=0.05)  # sets momentum to 0.05

    Notes:
        This is an in-place operation.

    Args:
        module: PyTorch `Module`.
        class_or_tuple: All instances of given `class_or_tuple` are to be tweaked.
        must_exist: If `True` an AttributeError is raised if keywords do not exist.
        recursive: Whether to search for modules recursively.
        **kwargs: Attributes to be tweaked: `<attribute_name>=<value>`.
    """
    for handle, key, mod in iter_submodules(module, class_or_tuple, recursive=recursive):
        for k, v in kwargs.items():
            if must_exist:
                getattr(mod, k)
            setattr(mod, k, v)


def replace_module_(module: nn.Module, class_or_tuple, substitute: Union[Type[nn.Module], nn.Module], recursive=True,
                    inherit_attr: Union[list, str, dict] = None, **kwargs):
    """Replace module.

    Replace all occurrences of `class_or_tuple` in `module` with `substitute`.

    Examples:
        >>> # Replace all ReLU activations with LeakyReLU
        ... cd.replace_module_(network, nn.ReLU, nn.LeakyReLU)

        >>> # Replace all BatchNorm layers with InstanceNorm and inherit `num_features` attribute
        ... cd.replace_module_(network, nn.BatchNorm2d, nn.InstanceNorm2d, inherit_attr=['num_features'])

        >>> # Replace all BatchNorm layers with GroupNorm and inherit `num_features` attribute
        ... cd.replace_module_(network, nn.BatchNorm2d, nn.GroupNorm, num_groups=32,
        ...                    inherit_attr={'num_channels': 'num_features'})

    Args:
        module: Module.
        class_or_tuple: Class or tuple of classes that are to be replaced.
        substitute: Substitute class or object.
        recursive: Whether to replace modules recursively.
        inherit_attr: Attributes to be inherited. String, list or dict of attribute names.
            Attribute values are retrieved from replaced module and passed to substitute constructor.
            Formats:
            ``'attr_name'``,
            ``['attr_name0', 'attr_name1', ...]``,
            ``{'substitute_kw0': 'attr_name0', ...}``
        **kwargs: Keyword arguments passed to substitute constructor if it is a class.
    """
    for handle, k, mod in iter_submodules(module, class_or_tuple, recursive=recursive):
        if isinstance(substitute, nn.Module):
            handle[k] = substitute
        else:
            kw = {}
            if isinstance(inherit_attr, str):
                inherit_attr = [inherit_attr]
            if isinstance(inherit_attr, list):
                kw = {k: mod.__dict__[k] for k in inherit_attr}
            elif isinstance(inherit_attr, dict):
                kw = {k: mod.__dict__[v] for k, v in inherit_attr.items()}
            handle[k] = substitute(**kwargs, **kw)


def inject_extra_repr_(module, name, fn):
    """Inject extra representation.

    Injects additional ``extra_repr`` function to ``module``.
    This can be helpful to indicate presence of hooks.

    Note:
        This is an inplace operation.

    Notes:
        - This op may impair pickling.

    Args:
        module: Module.
        name: Name of the injected function (only used to avoid duplicate injection).
        fn: Callback function.

    """

    def extra_repr(self=module):
        vals = [self.extra_repr_orig()] + list(f(self) for f in self.extra_repr_funcs.values())
        return ', '.join([v for v in vals if v])

    if not hasattr(module, 'extra_repr_orig'):
        module.extra_repr_orig = module.extra_repr
        module.extra_repr_funcs = {}
        module.extra_repr = extra_repr
    module.extra_repr_funcs[name] = fn


def wrap_module_(module: nn.Module, class_or_tuple, wrapper, recursive=True, **kwargs):
    for handle, k, mod in iter_submodules(module, class_or_tuple, recursive=recursive):
        handle[k] = wrapper(handle[k], **kwargs)


def spectral_norm_(module, class_or_tuple=nn.Conv2d, recursive=True, name='weight', add_repr=False, **kwargs):
    """Spectral normalization.

    Applies spectral normalization to parameters of all occurrences of ``class_or_tuple`` in the given module.

    Note:
        This is an inplace operation.

    References:
        - https://arxiv.org/pdf/1802.05957.pdf

    Args:
        module: Module.
        class_or_tuple: Class or tuple of classes whose parameters are to be normalized.
        recursive: Whether to search for modules recursively.
        name: Name of weight parameter.
        add_repr: Whether to indicate use of spectral norm in a module's representation.
            Note that this may impair pickling.
        **kwargs: Additional keyword arguments for ``torch.nn.utils.spectral_norm``.
    """

    def extra_repr(self):
        if 'torch.nn.utils.spectral_norm.SpectralNorm' in str(list(self._forward_pre_hooks.values())):
            return 'spectral_norm=True'

    for handle, k, mod in iter_submodules(module, class_or_tuple, recursive=recursive):
        if mod._parameters.get(name) is not None:
            handle[k] = nn.utils.spectral_norm(handle[k], name=name, **kwargs)
            if add_repr:
                inject_extra_repr_(handle[k], 'spectral_norm', extra_repr)


def weight_norm_(module, class_or_tuple=nn.Conv2d, recursive=True, name='weight', add_repr=False, **kwargs):
    """Weight normalization.

    Applies weight normalization to parameters of all occurrences of ``class_or_tuple`` in the given module.

    Note:
        This is an inplace operation.

    References:
        - https://proceedings.neurips.cc/paper/2016/file/ed265bc903a5a097f61d3ec064d96d2e-Paper.pdf

    Args:
        module: Module.
        class_or_tuple: Class or tuple of classes whose parameters are to be normalized.
        recursive: Whether to search for modules recursively.
        name: Name of weight parameter.
        add_repr: Whether to indicate use of weight norm in a module's representation.
            Note that this may impair pickling.
        **kwargs: Additional keyword arguments for ``torch.nn.utils.weight_norm``.
    """

    def extra_repr(self):
        if 'torch.nn.utils.weight_norm.WeightNorm' in str(list(self._forward_pre_hooks.values())):
            return 'weight_norm=True'

    for handle, k, mod in iter_submodules(module, class_or_tuple, recursive=recursive):
        if mod._parameters.get(name) is not None:
            handle[k] = nn.utils.weight_norm(handle[k], name=name, **kwargs)
            if add_repr:
                inject_extra_repr_(handle[k], 'weight_norm', extra_repr)


def get_device(module: Union[nn.Module, Tensor, torch.device]):
    """Get device.

    Get device from Module.

    Args:
        module: Module. If ``module`` is a string or ``torch.device`` already, it is returned as is.

    Returns:
        Device.
    """
    if isinstance(module, torch.device):
        return module
    elif isinstance(module, str):
        return module
    elif hasattr(module, 'device'):
        return module.device
    p: nn.parameter.Parameter = next(module.parameters())
    return p.device


def _params(module: nn.Module, trainable=None, recurse=True) -> Iterator[nn.Parameter]:
    return (p for p in module.parameters(recurse=recurse) if (trainable is None or p.requires_grad == trainable))


def trainable_params(module: nn.Module, recurse=True) -> Iterator[nn.Parameter]:
    """Trainable parameters.

    Retrieve all trainable parameters.

    Args:
        module: Module.
        recurse: Whether to also include parameters of all submodules.

    Returns:
        Module parameters.
    """
    return _params(module, True, recurse=recurse)


def frozen_params(module: nn.Module, recurse=True) -> Iterator[nn.Parameter]:
    """Frozen parameters.

    Retrieve all frozen parameters.

    Args:
        module: Module.
        recurse: Whether to also include parameters of all submodules.

    Returns:
        Module parameters.
    """
    return _params(module, False, recurse=recurse)


def num_params(module: nn.Module, trainable=None, recurse=True) -> int:
    """Number of parameters.

    Count the number of parameters.

    Args:
        module: Module
        trainable: Optionally filter for trainable or frozen parameters.
        recurse: Whether to also include parameters of all submodules.

    Returns:
        Number of parameters.
    """
    return sum(p.numel() for p in _params(module, trainable, recurse=recurse))


def count_submodules(module: nn.Module, class_or_tuple) -> int:
    """Count submodules.

    Count the number of submodules of the specified type(-es).

    Examples:
        >>> count_submodules(cd.models.U22(1, 0), nn.Conv2d)
        22

    Args:
        module: Module.
        class_or_tuple: All instances of given `class_or_tuple` are to be counted.

    Returns:
        Number of submodules.
    """
    return np.sum([1 for m in module.modules() if isinstance(m, class_or_tuple)])


def ensure_num_tuple(v, num=2, msg=''):
    if isinstance(v, (int, float)):
        v = (v,) * num
    elif isinstance(v, (list, tuple)):
        pass
    else:
        raise ValueError(msg)
    return v


class Bytes(int):
    """Bytes.

    Printable integer that represents Bytes.

    """
    UNITS = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB', 'BB']

    def __str__(self):
        n = np.log2(int(self)) if self > 0 else 0
        s = None
        for i, tag in enumerate(self.UNITS):
            if n < (i + 1) * 10 or i == len(self.UNITS) - 1:
                s = str(np.round(float(self) / (2 ** (10 * i)), 2)) + tag
                break
        return s

    __repr__ = __str__


class Percent(float):
    """Percent.

    Printable float that represents percentage.

    """

    def __str__(self):
        return '%g%%' % np.round(self, 2)

    __repr__ = __str__


class GpuStats:
    def __init__(self, delimiter=', '):
        """GPU Statistics.

        Simple interface to print live GPU statistics from ``pynvml``.

        Examples:
            >>> import celldetection as cd
            >>> stat = cd.GpuStats()  # initialize once
            >>> print(stat)  # print current statistics
            gpu0(free: 22.55GB, used: 21.94GB, util: 93%), gpu1(free: 1.03GB, used: 43.46GB, util: 98%)

        Args:
            delimiter: Delimiter used for printing.
        """
        try:
            nv.nvmlInit()
            self.num = nv.nvmlDeviceGetCount()
        except:
            self.num = 0
        self.delimiter = delimiter

    def __len__(self):
        return self.num

    def __getitem__(self, item: int):
        if item >= len(self):
            raise IndexError
        h = nv.nvmlDeviceGetHandleByIndex(item)
        idx = nv.nvmlDeviceGetIndex(h)
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        uti = nv.nvmlDeviceGetUtilizationRates(h)
        return idx, dict(
            free=Bytes(mem.free),
            used=Bytes(mem.used),
            util=Percent(uti.gpu)
        )

    def __str__(self):
        deli = self.delimiter
        return deli.join([f'gpu{i}({deli.join([f"{k}: {v}" for k, v in stat.items()])})' for i, stat in self])

    __repr__ = __str__


class Tiling:
    def __init__(self, tile_size: tuple, context_shape: tuple, overlap=0):
        self.overlap = overlap
        self.tile_size = tuple(tile_size)
        self.context_size = context_shape[:len(self.tile_size)]
        self.num_tiles_per_dim = np.ceil(np.array(self.context_size) / np.array(self.tile_size)).astype('int')
        self.num_tiles = np.prod(self.num_tiles_per_dim)

    def __len__(self):
        return self.num_tiles

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError
        tile_index = np.unravel_index(item, shape=self.num_tiles_per_dim)
        start = tile_index * np.array(self.tile_size)
        stop = np.minimum(start + self.tile_size, self.context_size)
        start_wo = np.maximum(start - self.overlap, 0)
        stop_wo = np.minimum(stop + self.overlap, self.context_size)
        start_ex = start - start_wo
        stop_ex = start - start_wo + stop - start
        return dict(
            start=start,
            stop=stop,
            slices=tuple([slice(a, b) for a, b in zip(start, stop)]),
            slices_with_overlap=tuple([slice(a, b) for a, b in zip(start_wo, stop_wo)]),
            slices_to_remove_overlap=tuple([slice(a, b) for a, b in zip(start_ex, stop_ex)]),
            start_ex=start_ex,
            stop_ex=stop_ex,
            start_with_overlap=start_wo,
            stop_with_overlap=stop_wo,
            num_tiles=self.num_tiles,
            num_tiles_per_dim=self.num_tiles_per_dim
        )


def to_h5(filename, mode='w', chunks=False, compression=None, overwrite=False, create_dataset_kw: dict = None,
          **kwargs):
    """To hdf5 file.

    Write data to hdf5 file.

    Args:
        filename: File name.
        mode: Mode.
        chunks: Chunks setting for created datasets. Chunk shape, or True to enable auto-chunking.
        compression: Compression setting for created datasets. Legal values are 'gzip', 'szip', 'lzf'. If an integer
            in range(10), this indicates gzip compression level. Otherwise, an integer indicates the number of a
            dynamically loaded compression filter.
        overwrite: Whether to overwrite existing dataset.
        create_dataset_kw: Additional keyword arguments for ``h5py.File().create_dataset``.
        **kwargs: Data as ``{dataset_name: data}``.

    """
    create_dataset_kw = {} if create_dataset_kw is None else create_dataset_kw
    with h5py.File(filename, mode) as h:
        for k, v in kwargs.items():
            exists = k in h
            if overwrite and exists:
                del h[k]
            elif exists:
                h[k][:] = v
            else:
                h.create_dataset(k, data=v, compression=compression, chunks=chunks, **create_dataset_kw)


def to_tiff(filename, image, mode='w', method='tile', bigtiff=True):
    """To tiff file.

    Write ``image`` to tiff file using ``pytiff``.
    By default, the tiff is tiled, s.t. crops can be read from disk without loading the entire image into memory first.

    Notes:
        - ``pytiff`` must be installed to use this function.

    References:
        https://pytiff.readthedocs.io/en/master/quickstart.html

    Args:
        filename: File name.
        image: Image.
        mode: Mode.
        method: Method. Either ``'tile'`` or ``'scanline'``.
        bigtiff: Whether to use bigtiff format.

    """
    try:
        from pytiff import Tiff
    except ModuleNotFoundError:
        raise ModuleNotFoundError('To use the to_tiff function pytiff must be installed.\n'
                                  'See: https://pytiff.readthedocs.io/en/master/quickstart.html')
    with Tiff(filename, mode, bigtiff=bigtiff) as handle:
        handle.write(image, method=method)


def exponential_moving_average_(module_avg, module, alpha=.999, alpha_non_trainable=0., buffers=True):
    """Exponential moving average.

    Update the variables of ``module_avg`` to be slightly closer to ``module``.

    References:
        - https://arxiv.org/pdf/1710.10196.pdf
        - https://arxiv.org/pdf/2006.07733.pdf

    Notes:
        - Whether a parameter is trainable or not is checked on ``module``
        - ``module_avg`` can be on different device and entirely frozen

    Args:
        module_avg: Average module. The parameters of this model are to be updated.
        module: Other Module.
        alpha: Fraction of trainable parameters of ``module_avg``; (1 - alpha) is fraction of trainable
            parameters of ``module``.
        alpha_non_trainable: Same as ``alpha``, but for non-trainable parameters.
        buffers: Whether to copy buffers from ``module`` to ``module_avg``.
    """
    device = get_device(module_avg)
    with torch.no_grad():
        for avg, new in zip(_params(module_avg), _params(module)):
            a = alpha if new.requires_grad else alpha_non_trainable
            avg.data.mul_(a).add_(new.data.to(device), alpha=1 - a)
    if buffers:
        for avg, new in zip(module_avg.buffers(), module.buffers()):
            avg.copy_(new)


def to_json(filename, obj, mode='w'):
    """To JSON.

    Dump ``obj`` to JSON file with name ``filename``.

    Args:
        filename: File name.
        obj: Object.
        mode: File mode.
    """
    with open(filename, mode) as fp:
        json.dump(obj, fp)


def from_json(filename):
    """From JSON.

    Load object from JSON file with name ``filename``.

    Args:
        filename: File name.
    """
    with open(filename, 'r') as fp:
        v = json.load(fp)
    return v


def get_nd_conv(dim: int):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return getattr(nn, 'Conv%dd' % dim)


def get_nd_max_pool(dim: int):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return getattr(nn, 'MaxPool%dd' % dim)


def get_nd_batchnorm(dim: int):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return getattr(nn, 'BatchNorm%dd' % dim)


def get_nd_dropout(dim: int):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return getattr(nn, 'Dropout%dd' % dim)


def get_nd_linear(dim: int):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return ['', 'bi', 'tri'][dim - 1] + 'linear'
