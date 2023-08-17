"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references and DETR's official implementation.
"""
import torch
from PIL import Image

from src import utils
from .NestedTensor import NestedTensor

log = utils.get_pylogger(__name__)


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = NestedTensor.nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def unnorm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = torch.tensor(std, device=img.device).reshape(3, 1, 1)
    mean = torch.tensor(mean, device=img.device).reshape(3, 1, 1)
    return img * std + mean


def load_pretrained(model, checkpoint, drop_prefix=None):
    model_dict = model.state_dict()
    model_weights = (
        checkpoint["model"] if "model" in checkpoint else checkpoint["state_dict"]
    )

    # Check if shapes between model_dict and checkpoint match, otherwise add to new_state_dict
    new_state_dict = {}
    for k, v in model_weights.items():
        # Remove prefix if needed
        if drop_prefix is not None and k.startswith(drop_prefix):
            k = k[len(drop_prefix) :]

        if k in model_dict and model_dict[k].shape == v.shape:
            new_state_dict[k] = v
        elif k in model_dict and model_dict[k].shape != v.shape:
            log.warning(
                f"Skipping {k} from pretrained weights: shape mismatch ({v.shape} vs {model_dict[k].shape})"
            )
        else:
            log.warning(f"Skipping {k} from pretrained weights: not found in model")

    log.info(f"Total weights from file: {len(model_weights)}")
    log.info(f"Total weights loaded: {len(new_state_dict)}")

    model_dict.update(new_state_dict)
    log.info(model.load_state_dict(model_dict, strict=False))

def get_annotation_id(annotation_name, face_only=False):
    # NOTE: yikes, can we do better?
    return list(get_annotations().keys())[
        list(get_annotations().values()).index(annotation_name)
    ]


def get_annotations():
    return {
        0: "face",
        1: "no-object",
    }


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
