# Character-Consistent Image Generation with IP-Adapter FaceID

A production ComfyUI workflow that generates a consistent character — same face, free pose — from a single reference image, engineered to run on a 4GB consumer GPU (GTX 1650). Built as the single-frame foundation for character consistency in AI video.

## Problem

Character consistency is the central unsolved problem in AI-generated media. A text-to-image model given "the same person in five scenes" produces five different people. For video, this is fatal: a character's face drifting frame to frame breaks the illusion entirely.

The naive fixes are expensive: train a LoRA or DreamBooth model per character (hours of compute, a dataset each), or accept inconsistency. Neither scales for production work where you need a known character on demand.

A second, hardware constraint: doing this at all on a 4GB GPU, without renting cloud compute.

## Solution

IP-Adapter FaceID injects facial identity at the model's cross-attention level from a **single reference image** — no per-character training. It uses insightface to extract a pure facial-identity vector and discards the reference's composition, so the **face stays locked while pose, scene, and framing remain fully controllable by the text prompt**.

This was a deliberate second iteration. The first build used the standard IP-Adapter, which transferred the reference's entire composition and locked every output into the same frontal pose. Diagnosing that limitation and moving to the FaceID architecture is the core engineering decision of the project.


## Workflow Architecture

![Workflow Architecture](images/workflow\_diagram.png)



*The MODEL is the carrier: Checkpoint → Unified Loader FaceID → IPAdapter FaceID → KSampler. The face reference is converted to an identity vector by insightface and injected into the model. The text prompt independently controls pose, scene, and style.*



Load Checkpoint ─MODEL──> Unified Loader FaceID ─MODEL──> IPAdapter FaceID ─MODEL──> KSampler ──> VAE Decode ──> Save

(SD 1.5)                  │  (model+LoRA+insightface)        ▲                        ▲

&#x20;                           └──────────IPADAPTER────────────────┤                        │



Load Image (reference) ─────────────IMAGE────────────────────> IPAdapter FaceID          │

CLIP Text Encode (pos/neg) ─────────CONDITIONING─────────────────────────────────────────┤

Empty Latent (512x512) ─────────────LATENT──────────────────────────────────────────────┘

The MODEL object is the carrier. The reference image is converted to a face identity vector and injected into it; the text prompt conditions everything else. Identity lives in the model, not the output image — which is why it survives pose and scene changes.

## Technical Components

|Component|Choice|Rationale|
|-|-|-|
|Base model|Realistic Vision (SD 1.5, fp16)|Fits \~2GB VRAM; SDXL needs 6–8GB.|
|Identity|IP-Adapter FaceID Plus v2 (SD 1.5)|Extracts identity vector only; frees pose.|
|Face detection|insightface (buffalo\_l)|Detects and encodes the reference face.|
|Paired LoRA|faceid-plusv2 SD 1.5 LoRA|Required companion to the v2 model.|
|Encoder|CLIP ViT-H|Required by the IP-Adapter stack.|
|Resolution|512×512|SD 1.5 native; 4GB-safe.|
|Sampler|dpmpp\_2m + Karras, 25 steps|Best quality-per-step on SD 1.5.|
|Memory|`--lowvram --fp32-vae`|Offload to RAM; fp32 VAE avoids 16-series black images.|

Measured peak VRAM: \~2.6GB of 4GB during sampling.

## Results

*(Insert a 2×3 grid: reference image, then the same character generated in multiple poses and scenes — frontal, walking, side profile, different lighting. This grid is the proof of consistency and the centerpiece of the project.)*

The face remains recognizably consistent across generations while pose, background, and framing vary with the prompt — the behavior the standard IP-Adapter could not achieve.

## Engineering Notes (debugging journey)

Three failures, each diagnosed to root cause:

* **Black image output.** Pipeline ran clean with no error; only the final image was black. Isolated to the VAE decode step — fp16 numerical instability on the GTX 1650 (Turing) producing NaNs. Fixed with `--fp32-vae`.
* **Pose lock.** Standard IP-Adapter reproduced the reference's frontal composition at every weight. Confirmed by lowering weight (background/hair varied, pose didn't), then resolved architecturally by switching to FaceID.
* **Model/LoRA version mismatch.** FaceID Plus v2 model paired with a v1 LoRA threw "LoRA not found." Matched the pair (both plusv2) to resolve.

These are included deliberately: the value of the project is the diagnostic reasoning under a hard hardware constraint, not just a working graph.

## Future Improvements

* **Toward video:** extend with AnimateDiff, or feed consistent frames into a video diffusion model — the direct path from consistent stills to a consistent character across frames.
* **Explicit pose control:** add ControlNet (OpenPose) to direct pose rather than relying on the prompt.
* **Higher fidelity:** move to SDXL FaceID on a larger GPU for finer detail.
* **Multi-character:** attention masking with a second FaceID branch.

\---

Built and tested on consumer hardware (GTX 1650, 4GB VRAM, 8GB RAM). The engineering focus is identity-consistent generation under a strict memory budget — and the foundation for solving character consistency in AI video.

