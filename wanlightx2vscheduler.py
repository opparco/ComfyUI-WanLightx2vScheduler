import torch
import comfy.sample
import comfy.samplers
import comfy.utils
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import latent_preview


class WanLightx2vSchedulerBasic:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
            "sigma_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5000.0, "step": 0.01, "round": False}),
            "sigma_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5000.0, "step": 0.01, "round": False}),
            "shift": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1, "round": False}), }}

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, sigma_max, sigma_min, shift):
        t = torch.linspace(1.0, 0.0, steps + 1)
        t_shift = shift * t / (1.0 + (shift - 1.0) * t)
        sigmas = torch.as_tensor(sigma_min + (sigma_max - sigma_min) * t_shift, dtype=torch.float32)
        print("SIGMAS:", sigmas)
        return (sigmas,)


class WanLightx2vSchedulerBasicFromModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
            "shift": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1, "round": False}), }}

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, shift):
        model_sampling = model.get_model_object("model_sampling")
        sigma_min = float(model_sampling.sigma_min)
        sigma_max = float(model_sampling.sigma_max)
        t = torch.linspace(1.0, 0.0, steps + 1)
        t_shift = shift * t / (1.0 + (shift - 1.0) * t)
        sigmas = torch.as_tensor(sigma_min + (sigma_max - sigma_min) * t_shift, dtype=torch.float32)
        print("SIGMAS:", sigmas)
        return (sigmas,)


class KSamplerAdvancedPartialSigmas(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {"required": {
            "model": (IO.MODEL, {}),
            "positive": (IO.CONDITIONING, {}),
            "negative": (IO.CONDITIONING, {}),
            "latent_image": (IO.LATENT, {}),
            # "sampler": (IO.SAMPLER, {}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
            "sigmas": (IO.SIGMAS, {}),
            "cfg": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "start_at_step": (IO.INT, {"default": 0, "min": 0, "max": 100000}),
            "end_at_step": (IO.INT, {"default": 4, "min": 1, "max": 100001}),
            "add_noise": (IO.BOOLEAN, {"default": True}),
            "noise_seed": (IO.INT, {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}), }}

    RETURN_TYPES = (IO.LATENT, IO.LATENT)
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, positive, negative, latent_image, sampler_name, sigmas, cfg, start_at_step, end_at_step, add_noise, noise_seed):
        total_steps = max(sigmas.shape[-1] - 1, 0)
        start = max(0, min(int(start_at_step), total_steps))
        end = max(start + 1, min(int(end_at_step) + 1, total_steps + 1))
        sigmas_slice = sigmas[start:end]
        print("SIGMAS slice:", sigmas_slice)

        latent = latent_image.copy()
        x = comfy.sample.fix_empty_latent_channels(model, latent["samples"])
        latent["samples"] = x

        if add_noise:
            batch_inds = latent.get("batch_index", None)
            noise = comfy.sample.prepare_noise(x, noise_seed, batch_inds)
        else:
            noise = torch.zeros(x.shape, dtype=x.dtype, layout=x.layout, device="cpu")

        noise_mask = latent.get("noise_mask", None)
        x0_output = {}
        callback = latent_preview.prepare_callback(model, sigmas_slice.shape[-1] - 1, x0_output)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        sampler = comfy.samplers.sampler_object(sampler_name)

        samples = comfy.sample.sample_custom(
            model=model,
            noise=noise,
            cfg=float(cfg),
            sampler=sampler,
            sigmas=sigmas_slice,
            positive=positive,
            negative=negative,
            latent_image=x,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise_seed,
        )

        out = latent.copy()
        out["samples"] = samples

        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out

        return (out, out_denoised)

# registration
NODE_CLASS_MAPPINGS = {
    "WanLightx2vSchedulerBasic": WanLightx2vSchedulerBasic,
    "WanLightx2vSchedulerBasicFromModel": WanLightx2vSchedulerBasicFromModel,
    "KSamplerAdvancedPartialSigmas": KSamplerAdvancedPartialSigmas,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerAdvancedPartialSigmas": "KSampler Advanced (Partial Sigmas)",
}
