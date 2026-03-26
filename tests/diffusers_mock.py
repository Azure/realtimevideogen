from unittest.mock import MagicMock

import importlib.util

from typing import Dict
from typing import Any
from typing import Tuple


def _make_pipeline_class(name: str) -> type:
    """Create a real class that inherits from MagicMock so it can be used with isinstance()."""
    cls = type(name, (MagicMock,), {})
    instance = cls()
    cls.from_pretrained = MagicMock(return_value=instance)  # type: ignore[attr-defined]
    return cls


class DiffusersMock(MagicMock):
    def __init__(
        self,
        *args: Tuple,
        **kwargs: Dict
    ) -> None:
        super().__init__(*args, **kwargs)

        self.__spec__ = importlib.util.spec_from_loader("diffusers", loader=None)

        # Pipeline / model classes as real types (usable with isinstance).
        self.FluxPipeline = _make_pipeline_class("FluxPipeline")
        self.Flux2Pipeline = _make_pipeline_class("Flux2Pipeline")
        self.Flux2KleinPipeline = _make_pipeline_class("Flux2KleinPipeline")
        self.FluxKontextPipeline = _make_pipeline_class("FluxKontextPipeline")
        self.FluxControlNetModel = _make_pipeline_class("FluxControlNetModel")
        self.FluxControlNetPipeline = _make_pipeline_class("FluxControlNetPipeline")
        self.DiffusionPipeline = _make_pipeline_class("DiffusionPipeline")
        self.HunyuanVideoFramepackPipeline = _make_pipeline_class("HunyuanVideoFramepackPipeline")
        self.AutoencoderKLHunyuanVideo = _make_pipeline_class("AutoencoderKLHunyuanVideo")
        self.FlowMatchEulerDiscreteScheduler = _make_pipeline_class("FlowMatchEulerDiscreteScheduler")
        self.LTXConditionPipeline = _make_pipeline_class("LTXConditionPipeline")
        self.LTXLatentUpsamplePipeline = _make_pipeline_class("LTXLatentUpsamplePipeline")
        self.QwenImagePipeline = _make_pipeline_class("QwenImagePipeline")
        self.QwenImageEditPipeline = _make_pipeline_class("QwenImageEditPipeline")
        self.CogView4Pipeline = _make_pipeline_class("CogView4Pipeline")
        self.HiDreamImagePipeline = _make_pipeline_class("HiDreamImagePipeline")

    def get_sub_modules(self) -> Dict[str, Any]:
        pipelines_mock = MagicMock()
        # Expose pipeline classes that are imported from diffusers.pipelines
        pipelines_mock.FluxControlNetPipeline = self.FluxControlNetPipeline
        pipelines_mock.pipeline_utils = MagicMock()
        pipelines_mock.pipeline_utils.DiffusionPipeline = self.DiffusionPipeline

        return {
            "diffusers": self,
            "diffusers.models": MagicMock(),
            "diffusers.models.autoencoders": MagicMock(),
            "diffusers.models.autoencoders.autoencoder_kl": MagicMock(),
            "diffusers.models.activations": MagicMock(),
            "diffusers.models.attention": MagicMock(),
            "diffusers.models.attention_processor": MagicMock(),
            "diffusers.models.embeddings": MagicMock(),
            "diffusers.models.unets": MagicMock(),
            "diffusers.models.unets.unet_2d_blocks": MagicMock(),
            "diffusers.models.normalization": MagicMock(),
            "diffusers.models.transformers": MagicMock(),
            "diffusers.models.transformers.dual_transformer_2d": MagicMock(),
            "diffusers.models.transformers.transformer_2d": MagicMock(),
            "diffusers.models.transformers.transformer_flux2": MagicMock(),
            "diffusers.models.transformers.transformer_hunyuan_video": MagicMock(),
            "diffusers.models.downsampling": MagicMock(),
            "diffusers.models.upsampling": MagicMock(),
            "diffusers.models.resnet": MagicMock(),
            "diffusers.pipelines": pipelines_mock,
            "diffusers.pipelines.pipeline_utils": pipelines_mock.pipeline_utils,
            "diffusers.pipelines.ltx": MagicMock(),
            "diffusers.pipelines.ltx.pipeline_ltx_condition": MagicMock(),
            "diffusers.configuration_utils": MagicMock(),
            "diffusers.schedulers": MagicMock(),
            "diffusers.schedulers.scheduling_utils": MagicMock(),
            "diffusers.utils": MagicMock(),
            "diffusers.utils.torch_utils": MagicMock(),
        }
