#####
# Modified from https://github.com/huggingface/diffusers/blob/v0.29.1/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py
# PhotoMaker v2 @ TencentARC and MCG-NKU 
# Author: Zhen Li
#####

# Copyright 2024 TencentARC and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ImageProjection, MultiAdapter, T2IAdapter, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    USE_PEFT_BACKEND,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines import StableDiffusionXLAdapterPipeline
from diffusers.utils import _get_model_file
from safetensors import safe_open
from huggingface_hub.utils import validate_hf_hub_args

from . import (
    PhotoMakerIDEncoder, # PhotoMaker v1
    PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken, # PhotoMaker v2
)
from photomaker.identity_prompt_parser import extract_identity_prompt_map_clean
from photomaker.identity_control_attn import SpatialRoutingProcessor
from layered.masks import create_half_masks
from photomaker.identity_slot_unet import IdentitySlotUNet

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def _preprocess_adapter_image(image, height, width):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        image = [np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])) for i in image]
        image = [
            i[None, ..., None] if i.ndim == 2 else i[None, ...] for i in image
        ]  # expand [h, w] or [h, w, c] to [b, h, w, c]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        if image[0].ndim == 3:
            image = torch.stack(image, dim=0)
        elif image[0].ndim == 4:
            image = torch.cat(image, dim=0)
        else:
            raise ValueError(
                f"Invalid image tensor! Expecting image tensor with 3 or 4 dimension, but recive: {image[0].ndim}"
            )
    return image


class PhotoMakerStableDiffusionXLAdapterPipeline(StableDiffusionXLAdapterPipeline):

    @validate_hf_hub_args
    def load_photomaker_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        weight_name: str,
        subfolder: str = '',
        trigger_words: Optional[List[str]] = None,
        pm_version: str = 'v2',
        **kwargs,
    ):
        """
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            weight_name (`str`):
                The weight name NOT the path to the weight.

            subfolder (`str`, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.

            trigger_words (`List[str]`, optional):
                A list of trigger tokens, one per identity.
            :
                The trigger word is used to identify the position of class word in the text prompt, 
                and it is recommended not to set it as a common word. 
                This trigger word must be placed after the class word when used, otherwise, it will affect the performance of the personalized generation.           
        """

        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"id_encoder": {}, "lora_weights": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("id_encoder."):
                            state_dict["id_encoder"][key.replace("id_encoder.", "")] = f.get_tensor(key)
                        elif key.startswith("lora_weights."):
                            state_dict["lora_weights"][key.replace("lora_weights.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if keys != ["id_encoder", "lora_weights"]:
            raise ValueError("Required keys are (`id_encoder` and `lora_weights`) missing from the state dict.")

        if trigger_words is None:
            trigger_words = ["img1", "img2"]

        self.trigger_words = trigger_words
        self.num_identities = len(trigger_words)
        self.num_tokens = 2   # tokens per identity
        self.identity_token_indices = None
        self.identity_prompt_map = None
        self.processor_text_input_ids = None

        # load finetuned CLIP image encoder and fuse module here if it has not been registered to the pipeline yet
        print(f"Loading PhotoMaker {pm_version} components [1] id_encoder from [{pretrained_model_name_or_path_or_dict}]...")
        self.id_image_processor = CLIPImageProcessor()
        if pm_version == "v1": # PhotoMaker v1 
            id_encoder = PhotoMakerIDEncoder()
        elif pm_version == "v2": # PhotoMaker v2
            id_encoder = PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken()
        else:
            raise NotImplementedError(f"The PhotoMaker version [{pm_version}] does not support")

        id_encoder.load_state_dict(state_dict["id_encoder"], strict=True)
        id_encoder = id_encoder.to(self.device, dtype=self.unet.dtype)    
        self.id_encoder = id_encoder

        # load lora into models
        print(f"Loading PhotoMaker {pm_version} components [2] lora_weights from [{pretrained_model_name_or_path_or_dict}]")
        self.load_lora_weights(state_dict["lora_weights"], adapter_name="photomaker")

        # Add trigger word token
        if self.tokenizer is not None:
            num_added = self.tokenizer.add_tokens(self.trigger_words, special_tokens=True)
            if num_added > 0:
                self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        num_added_2 = self.tokenizer_2.add_tokens(self.trigger_words, special_tokens=True)
        if num_added_2 > 0:
            self.text_encoder_2.resize_token_embeddings(len(self.tokenizer_2))


        if self.text_encoder is not None:
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        self.text_encoder_2.resize_token_embeddings(len(self.tokenizer_2))


        

    def encode_prompt_with_trigger_word(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        ### Added args
        num_id_images: int = 1,
        class_tokens_mask: Optional[torch.LongTensor] = None,
    ):
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Find the token id of the trigger words
        active_triggers = self.trigger_words[:num_id_images]

        trigger_token_ids = [
            self.tokenizer_2.convert_tokens_to_ids(t)
            for t in active_triggers
        ]

        print("Trigger words:", active_triggers)
        print("\n==============================")
        print("🔎 MULTI-TRIGGER DEBUG")
        print("Trigger words:", active_triggers)
        print("Trigger token IDs:", trigger_token_ids)
        print("==============================\n")



        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                decoded = self.tokenizer_2.convert_ids_to_tokens(text_input_ids[0])
                print("🔎 Tokenized prompt:")
                print(decoded)
                
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    print(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                # --------------------------------------------------
                # CLEAN MULTI-TRIGGER PARSING (FINAL CORRECT VERSION)
                # --------------------------------------------------

                input_ids = text_input_ids[0].tolist()

                clean_input_ids = []
                class_tokens_mask = []
                active_slots = []

                identity_count = 0
                i = 0

                while i < len(input_ids):

                    token_id = input_ids[i]

                    if token_id in trigger_token_ids:

                        slot_index = trigger_token_ids.index(token_id)
                        active_slots.append(slot_index)

                        if len(clean_input_ids) == 0:
                            raise ValueError("Trigger word cannot appear at start of prompt.")

                        class_token = clean_input_ids[-1]

                        clean_input_ids.pop()
                        class_tokens_mask.pop()

                        for _ in range(self.num_tokens):
                            clean_input_ids.append(class_token)
                            class_tokens_mask.append(True)

                        identity_count += 1
                        i += 1
                        continue

                    clean_input_ids.append(token_id)
                    class_tokens_mask.append(False)
                    i += 1

                print("✅ Active identity slots:", active_slots)

                self.active_slots = active_slots



                # Truncate or pad
                max_len = tokenizer.model_max_length

                if len(clean_input_ids) > max_len:
                    clean_input_ids = clean_input_ids[:max_len]
                    class_tokens_mask = class_tokens_mask[:max_len]
                else:
                    pad_len = max_len - len(clean_input_ids)
                    clean_input_ids += [tokenizer.pad_token_id] * pad_len
                    class_tokens_mask += [False] * pad_len


                clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0)
                class_tokens_mask = torch.tensor(class_tokens_mask, dtype=torch.bool).unsqueeze(0)

                # 🔥 STORE TOKEN IDS FOR ROUTING DEBUG
                self.processor_text_input_ids = clean_input_ids[0].clone().detach()

                print("🔍 num_id_images:", num_id_images)
                print("🔍 self.num_tokens:", self.num_tokens)

                expected_identity_tokens = num_id_images * self.num_tokens

                actual_identity_tokens = class_tokens_mask.sum().item()
                print("\n🔎 Identity Token Debug:")
                print("Expected identity token slots:", expected_identity_tokens)
                print("Mask True count:", actual_identity_tokens)
                print("Mask True indices:", torch.tensor(class_tokens_mask[0]).nonzero().flatten().tolist())

                print("🔍 Mask True count:", actual_identity_tokens)


                prompt_embeds = text_encoder(clean_input_ids.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        class_tokens_mask = class_tokens_mask.to(device=device) # TODO: ignoring two-prompt case
        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, class_tokens_mask, clean_input_ids

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        adapter_conditioning_scale: Union[float, List[float]] = 1.0,
        adapter_conditioning_factor: float = 1.0,
        clip_skip: Optional[int] = None,
        # Added parameters (for PhotoMaker)
        input_id_images=None,
        id_pixel_values=None,
        start_merge_step: int = 20, # TODO: change to `style_strength_ratio` in the future
        class_tokens_mask: Optional[torch.LongTensor] = None,
        id_embeds: Optional[torch.FloatTensor] = None,
        prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
        identity_bboxes: Optional[torch.FloatTensor] = None,
        
        **kwargs,
    ):
        print("🔥 id_pixel_values received:", id_pixel_values.shape if id_pixel_values is not None else None)
        print("🔥 id_embeds received:", id_embeds.shape if id_embeds is not None else None)
        r"""
        Function invoked when calling the pipeline for generation.
        Only the parameters introduced by PhotoMaker are discussed here. 
        For explanations of the previous parameters in StableDiffusionXLControlNetPipeline, please refer to https://github.com/huggingface/diffusers/blob/v0.25.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py

        Args:
            input_id_images (`PipelineImageInput`, *optional*): 
                Input ID Image to work with PhotoMaker.
            class_tokens_mask (`torch.LongTensor`, *optional*):
                Pre-generated class token. When the `prompt_embeds` parameter is provided in advance, it is necessary to prepare the `class_tokens_mask` beforehand for marking out the position of class word.
            prompt_embeds_text_only (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds_text_only (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        print("🔥 id_pixel_values:", None if id_pixel_values is None else id_pixel_values.shape)
        print("🔥 id_embeds:", None if id_embeds is None else id_embeds.shape)
        print("🔥 CUSTOM PIPELINE FILE LOADED 🔥")
        height, width = self._default_height_width(height, width, image)
        device = self._execution_device
        
        use_adapter = True if image is not None else False
        print(f"Use adapter: {use_adapter} ｜ output size: {(height, width)}")
        if use_adapter:
            if isinstance(self.adapter, MultiAdapter):
                adapter_input = []

                for one_image in image:
                    one_image = _preprocess_adapter_image(one_image, height, width)
                    one_image = one_image.to(device=device, dtype=self.adapter.dtype)
                    adapter_input.append(one_image)
            else:
                adapter_input = _preprocess_adapter_image(image, height, width)
                adapter_input = adapter_input.to(device=device, dtype=self.adapter.dtype)

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        


        # 1. Check inputs. Raise error if not correct
        self.identity_bboxes = identity_bboxes

        if identity_bboxes is not None:
            identity_bboxes = identity_bboxes.to(device)


        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
        )
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
    
        #        
        if prompt_embeds is not None and class_tokens_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `class_tokens_mask` also have to be passed. Make sure to generate `class_tokens_mask` from the same tokenizer that was used to generate `prompt_embeds`."
            )
        
        # --------------------------------------------------
        # STEP 1: Convert normalized bboxes to latent coords
        # --------------------------------------------------

        if self.identity_bboxes is not None:
            num_ids = self.identity_bboxes.shape[0]
            latent_h = height // 8
            latent_w = width // 8

            base_masks = torch.zeros(
                (num_ids, latent_h, latent_w),
                device=device
            )

            for i, (x1, y1, x2, y2) in enumerate(self.identity_bboxes):

                # Convert normalized → latent grid
                x1_lat = int(x1 * latent_w)
                x2_lat = int(x2 * latent_w)
                y1_lat = int(y1 * latent_h)
                y2_lat = int(y2 * latent_h)

                # Clamp to valid bounds
                x1_lat = max(0, min(latent_w - 1, x1_lat))
                x2_lat = max(0, min(latent_w, x2_lat))
                y1_lat = max(0, min(latent_h - 1, y1_lat))
                y2_lat = max(0, min(latent_h, y2_lat))

                base_masks[i, y1_lat:y2_lat, x1_lat:x2_lat] = 1.0
            print("Base masks shape:", base_masks.shape)
            for i in range(base_masks.shape[0]):
                print(f"Mask {i} sum:", base_masks[i].sum().item())

            if base_masks.shape[0] > 1:
                print("Are masks equal:",
                    torch.allclose(base_masks[0], base_masks[1]))


            self.identity_base_masks = base_masks

        
       
          



        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        
        if id_embeds is not None:
            if id_embeds.dim() == 2:
                # shape = [N, 512]
                num_id_images = id_embeds.shape[0]
            elif id_embeds.dim() == 3:
                # shape = [B, N, 512]
                num_id_images = id_embeds.shape[1]
            else:
                raise ValueError("Unexpected id_embeds shape")
        else:
            num_id_images = 1
        # 🔥 IMPORTANT
        self.num_identities = num_id_images

        print("🔍 id_embeds shape:", id_embeds.shape)
        print("🔍 num_id_images:", num_id_images)


        
        (
            prompt_embeds, 
            _,
            pooled_prompt_embeds,
            _,
            class_tokens_mask,
            clean_input_ids,
        ) = self.encode_prompt_with_trigger_word(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_id_images=num_id_images,
            class_tokens_mask=class_tokens_mask,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self._clip_skip,
        )
        # Store clean input ids for routing processor
        self.processor_text_input_ids = clean_input_ids[0]

        # --------------------------------------------------
        # Compute identity_token_indices
        # --------------------------------------------------

        self.identity_token_indices = []

        if class_tokens_mask is not None:
            mask = class_tokens_mask[0]
            token_positions = mask.nonzero(as_tuple=False).flatten().tolist()

            tokens_per_identity = self.num_tokens

            for i in range(0, len(token_positions), tokens_per_identity):
                self.identity_token_indices.append(
                    token_positions[i:i + tokens_per_identity]
                )

        print("🔥 identity_token_indices:", self.identity_token_indices)

        # --------------------------------------------------
        # Compute identity_prompt_map
        # --------------------------------------------------

        print("\n========== TOKEN DEBUG (Adapter) ==========")

        decoded = self.tokenizer_2.convert_ids_to_tokens(
            self.processor_text_input_ids.tolist()
        )
        print("Decoded tokens:", decoded)

        trigger_ids = [
            self.tokenizer_2.convert_tokens_to_ids(t)
            for t in self.trigger_words[:self.num_identities]
        ]

        print("Trigger words:", self.trigger_words[:self.num_identities])
        print("Trigger token ids:", trigger_ids)

        print("===========================================\n")


        identity_prompt_map = extract_identity_prompt_map_clean(
            text_input_ids=self.processor_text_input_ids.tolist(),
            identity_token_indices=self.identity_token_indices,
        )


        self.identity_prompt_map = identity_prompt_map


        # 4. Encode input prompt without the trigger word for delayed conditioning
        # encode, remove trigger word token, then decode
        tokens_text_only = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        active_triggers = self.trigger_words[:num_id_images]
        trigger_token_ids = [
            self.tokenizer_2.convert_tokens_to_ids(t)
            for t in active_triggers
        ]

        tokens_text_only = [
            t for t in tokens_text_only
            if t not in trigger_token_ids
        ]

        prompt_text_only = self.tokenizer.decode(tokens_text_only, add_special_tokens=False)
        (
            prompt_embeds_text_only,
            negative_prompt_embeds,
            pooled_prompt_embeds_text_only, # TODO: replace the pooled_prompt_embeds with text only prompt
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt_text_only,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds_text_only,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds_text_only,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self._clip_skip,
        )

        # 5. Prepare identity tensors (already processed in CLI)

        if input_id_images is not None and id_pixel_values is None:
            print("🔥 Converting input_id_images → id_pixel_values")

            if not isinstance(input_id_images[0], torch.Tensor):
                id_pixel_values = self.id_image_processor(
                    input_id_images,
                    return_tensors="pt"
                ).pixel_values

            # 🔥 Get TRUE device + dtype from vision model
            vision_param = next(self.vision_model.parameters())
            device = vision_param.device
            dtype  = vision_param.dtype

            id_pixel_values = id_pixel_values.unsqueeze(0).to(
                device=device,
                dtype=dtype
            )

            print("vision_model device:", device)
            print("vision_model dtype:", dtype)
            print("id_pixel_values device:", id_pixel_values.device)
            print("id_pixel_values dtype:", id_pixel_values.dtype)

        # 6. Get the updated text embedding with the stacked ID embedding
        if id_pixel_values is not None and id_embeds is not None:

            # ---------------------------------------
            # Move to correct device/dtype
            # ---------------------------------------
            encoder_param = next(self.id_encoder.parameters())
            target_device = encoder_param.device
            target_dtype  = encoder_param.dtype

            id_pixel_values = id_pixel_values.to(
                device=target_device,
                dtype=target_dtype
            )

            id_embeds = id_embeds.to(
                device=target_device,
                dtype=target_dtype
            )

            
            # ---------------------------------------
            # 🔥 Normalize + Equalize + Scale Identity Embeddings
            # ---------------------------------------

            scale_factor = 45.0  # slightly lower than 35

            # 1️⃣ Normalize each identity token
            id_embeds = torch.nn.functional.normalize(id_embeds, dim=-1)

            # 2️⃣ Equalize identity energy across identities
            # id_embeds shape = [B, N_id, 512] or [N_id, 512]

            identity_norms = torch.norm(id_embeds, dim=-1, keepdim=True)

            # Mean norm across all identity tokens
            mean_norm = identity_norms.mean()

            id_embeds = id_embeds / (mean_norm + 1e-8)

            # 3️⃣ Apply global scale
            id_embeds = id_embeds * scale_factor

            # Debug
            print("🔎 Identity embedding norms after equalization:")
            for identity_i in range(id_embeds.shape[0]):
                for token_i in range(id_embeds.shape[1]):
                    norm = torch.norm(id_embeds[identity_i, token_i]).item()
                    print(f"   Identity {identity_i} Token {token_i} norm: {norm:.4f}")

            
           
            # ---------------------------------------
            # Inject
            # ---------------------------------------
            prompt_embeds = self.id_encoder(
               id_pixel_values,
                prompt_embeds,
                class_tokens_mask,
                id_embeds
            )

            # DEBUG HERE
            mask_indices = class_tokens_mask[0].nonzero().flatten().tolist()

            print("🔎 Injected prompt token norms:")
            for idx in mask_indices:
                norm = torch.norm(prompt_embeds[0, idx]).item()
                print(f"Token {idx} norm: {norm:.4f}")

        else:
            print("⚠️ Skipping ID injection (missing id_pixel_values or id_embeds)")


        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # 6.1 Get the ip adapter embedding
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 7. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 8. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        if use_adapter:
            if isinstance(self.adapter, MultiAdapter):
                adapter_state = self.adapter(adapter_input, adapter_conditioning_scale)
                for k, v in enumerate(adapter_state):
                    adapter_state[k] = v
            else:
                adapter_state = self.adapter(adapter_input)
                for k, v in enumerate(adapter_state):
                    adapter_state[k] = v * adapter_conditioning_scale
            if num_images_per_prompt > 1:
                for k, v in enumerate(adapter_state):
                    adapter_state[k] = v.repeat(num_images_per_prompt, 1, 1, 1)
            if self.do_classifier_free_guidance:
                for k, v in enumerate(adapter_state):
                    adapter_state[k] = torch.cat([v] * 2, dim=0)

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        


        # --------------------------------------------------
        # 🔥 OPTIONAL: ATTACH SPATIAL + ROUTING PROCESSOR
        # --------------------------------------------------

        if (
            self.enable_routing
            and hasattr(self, "identity_base_masks")
            and self.identity_base_masks is not None
        ):

            new_processors = {}

            for name, proc in self.unet.attn_processors.items():

                if "attn2" in name:  # cross-attention only

                    new_processors[name] = SpatialRoutingProcessor(
                        identity_token_indices=self.identity_token_indices,
                        identity_prompt_map=self.identity_prompt_map,
                        base_masks=self.identity_base_masks,
                        tokenizer=self.tokenizer,
                        text_input_ids=self.processor_text_input_ids,

                        # --- Routing strengths ---
                        identity_bias=2.5,              # own identity boost
                        spatial_strength=1.0,           # spatial emphasis
                        outside_suppress=1.5,           # suppress outside region
                        routing_strength=6.0,           # attribute boost
                        cross_identity_strength=6.0,    # suppress other identities
                    )

                else:
                    new_processors[name] = proc

            self.unet.set_attn_processor(new_processors)
            print("✅ SpatialRoutingProcessor attached.")

        else:
            print("🚫 Routing disabled. Using default attention processors.")
        
        # Compute warmup steps (required for callback logic)
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order,
            0
        )
       
        # ==================================================
        # 🔥 11. Denoising loop
        # ==================================================

        # Set identity data once before loop
        if isinstance(self.unet, IdentitySlotUNet):
            if id_embeds.dim() == 2:
                id_embeds = id_embeds.unsqueeze(0)

            self.unet.set_identity_data(
                embeddings=id_embeds,
                bboxes=self.identity_bboxes
            )

        z_A = latents.clone()
        z_B = latents.clone()

        mask_A, mask_B = create_half_masks(z_A)
        mask_A = mask_A.to(dtype=z_A.dtype, device=z_A.device)
        mask_B = mask_B.to(dtype=z_A.dtype, device=z_A.device)


        # --------------------------------------------------
        # 🔎 Identity Influence Logging Buffers
        # --------------------------------------------------

        identity_A_strength = []
        identity_B_strength = []
        identity_ratio = []

        identity_A_spatial = []
        identity_B_spatial = []
        if isinstance(self.unet, IdentitySlotUNet):
                    self.unet.debug_similarity = []

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                # 🔥 Set current step here
                if isinstance(self.unet, IdentitySlotUNet):
                    self.unet.current_step = i
                # --------------------------------------------------
                # 🔥 1️⃣ DELAYED MERGE LOGIC (FIRST)
                # --------------------------------------------------

                if i <= start_merge_step:
                    current_prompt_embeds = torch.cat(
                        [negative_prompt_embeds, prompt_embeds_text_only], dim=0
                    ) if self.do_classifier_free_guidance else prompt_embeds_text_only

                    add_text_embeds = torch.cat(
                        [negative_pooled_prompt_embeds, pooled_prompt_embeds_text_only], dim=0
                    ) if self.do_classifier_free_guidance else pooled_prompt_embeds_text_only
                else:
                    current_prompt_embeds = torch.cat(
                        [negative_prompt_embeds, prompt_embeds], dim=0
                    ) if self.do_classifier_free_guidance else prompt_embeds

                    add_text_embeds = torch.cat(
                        [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
                    ) if self.do_classifier_free_guidance else pooled_prompt_embeds

                # --------------------------------------------------
                # 2️⃣ ADAPTER CONDITIONING
                # --------------------------------------------------

                if i < int(num_inference_steps * adapter_conditioning_factor) and use_adapter:
                    down_intrablock_additional_residuals = [
                        state.clone() for state in adapter_state
                    ]
                else:
                    down_intrablock_additional_residuals = None

                # --------------------------------------------------
                # 3️⃣ BUILD added_cond_kwargs
                # --------------------------------------------------

                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }

                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds

                # --------------------------------------------------
                # ==========================================================
                # 🔎 TEXT-ONLY BASELINES (FOR IDENTITY ATTRIBUTION)
                # ==========================================================

                # 1️⃣ Disable identity slots for pure text baseline
                if isinstance(self.unet, IdentitySlotUNet):
                    self.unet.set_active_slots([])

                # Build CFG embeddings for text-only prompt
                if self.do_classifier_free_guidance:
                    text_only_embeds_cfg = torch.cat(
                        [negative_prompt_embeds, prompt_embeds_text_only], dim=0
                    )
                else:
                    text_only_embeds_cfg = prompt_embeds_text_only

                # ----------------------------------------------------------
                # 🔹 Baseline for Stream A (uses z_A)
                # ----------------------------------------------------------

                latent_text_input_A = (
                    torch.cat([z_A] * 2) if self.do_classifier_free_guidance else z_A
                )
                latent_text_input_A = self.scheduler.scale_model_input(
                    latent_text_input_A, t
                )

                noise_text_only_A = self.unet(
                    latent_text_input_A,
                    t,
                    encoder_hidden_states=text_only_embeds_cfg,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond_A_to, noise_text_A_to = noise_text_only_A.chunk(2)
                    noise_text_only_A = (
                        noise_uncond_A_to
                        + guidance_scale * (noise_text_A_to - noise_uncond_A_to)
                    )

                # ----------------------------------------------------------
                # 🔹 Baseline for Stream B (uses z_B)
                # ----------------------------------------------------------

                latent_text_input_B = (
                    torch.cat([z_B] * 2) if self.do_classifier_free_guidance else z_B
                )
                latent_text_input_B = self.scheduler.scale_model_input(
                    latent_text_input_B, t
                )

                noise_text_only_B = self.unet(
                    latent_text_input_B,
                    t,
                    encoder_hidden_states=text_only_embeds_cfg,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond_B_to, noise_text_B_to = noise_text_only_B.chunk(2)
                    noise_text_only_B = (
                        noise_uncond_B_to
                        + guidance_scale * (noise_text_B_to - noise_uncond_B_to)
                    )

                # ==========================================================
                # 🔥 IMPORTANT: DO NOT ACTIVATE SLOTS HERE
                # Slots will be set properly before Stream A and Stream B
                # ==========================================================


                # --------------------------------------------------
                # STREAM-SPECIFIC TOKEN MASKING
                # --------------------------------------------------

                prompt_A = current_prompt_embeds.clone()
                prompt_B = current_prompt_embeds.clone()

                if hasattr(self, "identity_token_indices") and len(self.identity_token_indices) >= 2:
                    identity_A_indices = self.identity_token_indices[0]
                    identity_B_indices = self.identity_token_indices[1]

                    prompt_A[:, identity_B_indices, :] *= 0.0
                    prompt_B[:, identity_A_indices, :] *= 0.0
                
                
                # --------------------------------------------------
                # STREAM A
                # --------------------------------------------------

                if isinstance(self.unet, IdentitySlotUNet):
                    self.unet.set_active_slots([0])

                latent_A_input = torch.cat([z_A] * 2) if self.do_classifier_free_guidance else z_A
                latent_A_input = self.scheduler.scale_model_input(latent_A_input, t)

                noise_pred_A = self.unet(
                    latent_A_input,
                    t,
                    encoder_hidden_states=prompt_A,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                # 🔬 DEBUG HERE
                if isinstance(self.unet, IdentitySlotUNet):
                    print("Similarity length after Stream A:",
                        len(self.unet.debug_similarity))
                

                if self.do_classifier_free_guidance:
                    noise_uncond_A, noise_text_A = noise_pred_A.chunk(2)
                    noise_pred_A = noise_uncond_A + guidance_scale * (noise_text_A - noise_uncond_A)

                # --------------------------------------------------
                # STREAM B
                # --------------------------------------------------

                if isinstance(self.unet, IdentitySlotUNet):
                    self.unet.set_active_slots([1])

                latent_B_input = torch.cat([z_B] * 2) if self.do_classifier_free_guidance else z_B
                latent_B_input = self.scheduler.scale_model_input(latent_B_input, t)

                noise_pred_B = self.unet(
                    latent_B_input,
                    t,
                    encoder_hidden_states=prompt_B,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond_B, noise_text_B = noise_pred_B.chunk(2)
                    noise_pred_B = noise_uncond_B + guidance_scale * (noise_text_B - noise_uncond_B)


                # --------------------------------------------------
                # 🔎 IDENTITY INFLUENCE COMPUTATION
                # --------------------------------------------------

                # Compute identity residuals
                residual_A = noise_pred_A - noise_text_only_A
                residual_B = noise_pred_B - noise_text_only_B



                # Global magnitude
                A_strength = torch.norm(residual_A).item()
                B_strength = torch.norm(residual_B).item()

                # Optional: relative contribution to final noise
                total_norm = torch.norm(mask_A * noise_pred_A + mask_B * noise_pred_B).item() + 1e-8
                ratio_A = A_strength / total_norm
                ratio_B = B_strength / total_norm

                # Optional: spatial-only influence
                A_spatial = torch.norm(mask_A * residual_A).item()
                B_spatial = torch.norm(mask_B * residual_B).item()

                # Log values
                identity_A_strength.append(A_strength)
                identity_B_strength.append(B_strength)
                identity_ratio.append((ratio_A, ratio_B))
                identity_A_spatial.append(A_spatial)
                identity_B_spatial.append(B_spatial)
                # --------------------------------------------------
                # BLEND NOISE
                # --------------------------------------------------

                noise_pred = mask_A * noise_pred_A + mask_B * noise_pred_B

                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False
                )[0]

                z_A = latents.clone()
                z_B = latents.clone()
                # --------------------------------------------------
                # --------------------------------------------------
                # 🔎 DEBUG IMAGE AT start_merge_step + MID PHASE
                # 🔬 Identity Similarity-Based Bounding Box (UNet space)
                # --------------------------------------------------

                mid_step = int(num_inference_steps * 0.5)

                if i == start_merge_step or i == mid_step:

                    phase_name = (
                        "start_merge_step"
                        if i == start_merge_step
                        else "mid_phase"
                    )

                    print(f"\n🧪 Decoding image at {phase_name} = {i}")

                    debug_latents = latents.clone()

                    # -----------------------------
                    # Safe VAE decode
                    # -----------------------------
                    needs_upcasting = (
                        self.vae.dtype == torch.float16
                        and self.vae.config.force_upcast
                    )

                    if needs_upcasting:
                        self.upcast_vae()
                        debug_latents = debug_latents.to(
                            next(iter(self.vae.post_quant_conv.parameters())).dtype
                        )

                    decoded = self.vae.decode(
                        debug_latents / self.vae.config.scaling_factor,
                        return_dict=False
                    )[0]

                    debug_images = self.image_processor.postprocess(
                        decoded,
                        output_type="pil"
                    )

                    debug_image = debug_images[0]

                    # --------------------------------------------------
                    # 🔬 GET ARGMAX IDENTITY SEGMENTATION
                    # --------------------------------------------------

                    import numpy as np
                    from PIL import Image
                    import torch.nn.functional as F

                    if (
                        hasattr(self.unet, "debug_similarity")
                        and len(self.unet.debug_similarity) > 0
                    ):

                        similarity_entry = self.unet.debug_similarity[-1]

                        if "scores" not in similarity_entry:
                            print("⚠️ No full score tensor found (check _inject logging).")

                        else:
                            scores = similarity_entry["scores"]  # [B, N, H, W]

                            # Remove batch dimension
                            scores = scores[0]  # [N, H, W]

                            if scores.shape[0] < 2:
                                print("⚠️ Only one identity present — argmax segmentation meaningless.")
                            else:
                                # ---------------------------------
                                # 1️⃣ Argmax over identities
                                # ---------------------------------
                                identity_map = torch.argmax(scores, dim=0)  # [H, W]

                                # ---------------------------------
                                # 2️⃣ Upsample to image resolution
                                # ---------------------------------
                                identity_map = identity_map.unsqueeze(0).unsqueeze(0).float()

                                img_w, img_h = debug_image.size

                                identity_map_up = F.interpolate(
                                    identity_map,
                                    size=(img_h, img_w),
                                    mode="nearest"
                                )[0, 0].long()  # [H_img, W_img]

                                identity_np = identity_map_up.cpu().numpy()

                                # ---------------------------------
                                # 3️⃣ Define identity colors
                                # ---------------------------------
                                colors = [
                                    (255, 0, 0),    # Identity 0 → Red
                                    (0, 0, 255),    # Identity 1 → Blue
                                    (0, 255, 0),    # Identity 2 → Green
                                    (255, 255, 0),  # Identity 3 → Yellow
                                ]

                                overlay = np.zeros((img_h, img_w, 3), dtype=np.uint8)

                                num_ids = scores.shape[0]

                                for i_id in range(min(num_ids, len(colors))):
                                    overlay[identity_np == i_id] = colors[i_id]

                                overlay_img = Image.fromarray(overlay)

                                # ---------------------------------
                                # 4️⃣ Blend overlay with decoded image
                                # ---------------------------------
                                debug_image = Image.blend(
                                    debug_image.convert("RGBA"),
                                    overlay_img.convert("RGBA"),
                                    alpha=0.4
                                )

                                print("✅ Argmax identity segmentation generated")

                    else:
                        print("⚠️ No similarity map available from UNet")

                    # --------------------------------------------------
                    # Save
                    # --------------------------------------------------

                    debug_path = (
                        f"/teamspace/studios/this_studio/PhotoMaker/Data/Output/"
                        f"debug_{phase_name}_step_{i}.png"
                    )

                    debug_image.save(debug_path)

                    print("✅ Saved debug image:", debug_path)
                # --------------------------------------------------
                # CALLBACK (MUST BE INSIDE LOOP)
                # --------------------------------------------------

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and
                    (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)


        # --------------------------------------------------
        # 🔥 CLEAR IDENTITY STATE AFTER LOOP
        # --------------------------------------------------

        if isinstance(self.unet, IdentitySlotUNet):
            self.unet.clear_identity_data()
        # --------------------------------------------------
        # 🔎 PRINT IDENTITY INFLUENCE CURVES
        # --------------------------------------------------

        print("\n--- Identity Influence Summary ---")
        for i in range(len(identity_A_strength)):
            print(
                f"Step {i}: "
                f"A={identity_A_strength[i]:.2f} "
                f"B={identity_B_strength[i]:.2f} "
                f"RatioA={identity_ratio[i][0]:.3f} "
                f"RatioB={identity_ratio[i][1]:.3f}"
            )

       
        

                
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)