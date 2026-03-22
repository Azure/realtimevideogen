import torch
import librosa
import numpy as np

from typing import Dict
from typing import Tuple
from typing import Any

from PIL import Image
from einops import rearrange

from transformers import CLIPImageProcessor
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage


def get_audio_feature(
    feature_extractor: Any,
    audio_path: str
) -> Tuple[torch.Tensor, int]:
    audio_input, sampling_rate = librosa.load(audio_path, sr=16000)
    assert sampling_rate == 16000

    audio_features = []
    window = 750 * 640
    for i in range(0, len(audio_input), window):
        audio_feature = feature_extractor(audio_input[i:i + window],
                                          sampling_rate=sampling_rate,
                                          return_tensors="pt",
                                          ).input_features
        audio_features.append(audio_feature)

    audio_features = torch.cat(audio_features, dim=-1)
    return audio_features, len(audio_input) // 640


class VideoAudioTextLoaderVal():
    def __init__(
        self,
        image_size: int,
        text_encoder: Any,
        text_encoder_2: Any,
        feature_extractor: Any,
    ) -> None:
        self.image_size = image_size
        self.text_encoder = text_encoder  # llava_text_encoder
        self.text_encoder_2 = text_encoder_2  # clip_text_encoder
        self.feature_extractor = feature_extractor

        self.llava_transform = transforms.Compose(
            [
                transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.4082107), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        self.clip_image_processor = CLIPImageProcessor()

        self.device = torch.device("cuda")
        self.weight_dtype = torch.float16

    @staticmethod
    def get_text_tokens(
        text_encoder: Any,
        description: str,
        dtype_encode: str = "video"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_inputs = text_encoder.text2tokens(description, data_type=dtype_encode)
        text_ids = text_inputs["input_ids"].squeeze(0)
        text_mask = text_inputs["attention_mask"].squeeze(0)
        return text_ids, text_mask

    def encode_data(
        self,
        ref_image: Any,
        audio_path: str,
        prompt: str,
        fps: int
    ) -> Dict[str, Any]:
        prompt = "Authentic, Realistic, Natural, High-quality, Lens-Fixed, " + prompt

        img_size = self.image_size
        # ref_image = Image.open(image_path).convert('RGB')

        # Resize reference image
        w, h = ref_image.size
        scale = img_size / min(w, h)
        new_w = round(w * scale / 64) * 64
        new_h = round(h * scale / 64) * 64

        if img_size == 704:
            img_size_long = 1216
        if new_w * new_h > img_size * img_size_long:
            import math
            scale = math.sqrt(img_size * img_size_long / w / h)
            new_w = round(w * scale / 64) * 64
            new_h = round(h * scale / 64) * 64

        ref_image = ref_image.resize((new_w, new_h), Image.LANCZOS)

        ref_image = np.array(ref_image)
        ref_image = torch.from_numpy(ref_image)

        audio_input, audio_len = get_audio_feature(self.feature_extractor, audio_path)
        audio_prompts = audio_input[0]

        motion_bucket_id_heads = np.array([25] * 4)
        motion_bucket_id_exps = np.array([30] * 4)
        motion_bucket_id_heads = torch.from_numpy(motion_bucket_id_heads)
        motion_bucket_id_exps = torch.from_numpy(motion_bucket_id_exps)
        fps_tensor = torch.from_numpy(np.array(fps))

        to_pil = ToPILImage()
        pixel_value_ref = rearrange(ref_image.clone().unsqueeze(0), "b h w c -> b c h w")   # (b c h w)

        pixel_value_ref_llava = [self.llava_transform(to_pil(image)) for image in pixel_value_ref]
        pixel_value_ref_llava = torch.stack(pixel_value_ref_llava, dim=0)
        pixel_value_ref_clip = self.clip_image_processor(
            images=Image.fromarray((pixel_value_ref[0].permute(1, 2, 0)).data.cpu().numpy().astype(np.uint8)),
            return_tensors="pt"
        ).pixel_values[0]
        pixel_value_ref_clip = pixel_value_ref_clip.unsqueeze(0)

        # Encode text prompts
        text_ids, text_mask = self.get_text_tokens(self.text_encoder, prompt)
        text_ids_2, text_mask_2 = self.get_text_tokens(self.text_encoder_2, prompt)

        # Output
        return {
            "text_prompt": prompt,
            "pixel_value_ref": pixel_value_ref.to(dtype=torch.float16),                 # for vae (1, 3, h, w)
            "pixel_value_ref_llava": pixel_value_ref_llava.to(dtype=torch.float16),     # for llava (1, 3, 336, 336)
            # for clip_image_encoder (1, 3, 244, 244)
            "pixel_value_ref_clip": pixel_value_ref_clip.to(dtype=torch.float16),
            "audio_prompts": audio_prompts.to(dtype=torch.float16),
            "motion_bucket_id_heads": motion_bucket_id_heads.to(dtype=text_ids.dtype),
            "motion_bucket_id_exps": motion_bucket_id_exps.to(dtype=text_ids.dtype),
            "fps": fps_tensor.to(dtype=torch.float16),
            "text_ids": text_ids.clone(),                                               # for llava_text_encoder
            "text_mask": text_mask.clone(),                                             # for llava_text_encoder
            "text_ids_2": text_ids_2.clone(),                                           # for clip_text_encoder
            "text_mask_2": text_mask_2.clone(),                                         # for clip_text_encoder
            "audio_len": audio_len,
            "audio_path": audio_path,
        }
