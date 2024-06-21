<h1 align="center"> 
Adversarial Illusions in Multi-Modal Embeddings </h1>

<p align="center"> <i>Tingwei Zhang, Rishi Jha, Eugene Bagdasaryan, and Vitaly Shmatikov</i></p>

TODO: call out changes to imagebind, audioclip, and diffjpeg!
TODO: data/imagenet/ILSVRC2012_devkit_t12.tar.gz
      data/imagenet/ILSVRC2012_img_val.tar

Multi-modal embeddings encode texts, images, sounds, videos, etc., into a single embedding space, aligning representations across different modalities (e.g., associate an image of a dog with a barking sound). In this paper, we show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an image or a sound, an adversary can perturb it to make its embedding close to an arbitrary, adversary-chosen input in another modality.

These attacks are cross-modal and targeted: the adversary is free to align any image and any sound with any target of his choice. Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks and modalities, enabling a wholesale compromise of current and future downstream tasks and modalities not available to the adversary. Using ImageBind and AudioCLIP embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, zero-shot classification, and audio retrieval.

We investigate transferability of illusions across different embeddings and develop a black-box version of our method that we use to demonstrate the first adversarial alignment attack on Amazon's commercial, proprietary Titan embedding. Finally, we analyze countermeasures and evasion attacks.

Paper link:
[https://arxiv.org/abs/2308.11804](https://arxiv.org/abs/2308.11804)

<img src="image/illusion.png" alt="drawing" width="600"/>

We ran everything on a 48GB A40.

# Installation
- Clone the repository and download the submodules, run  `git submodule update`
- Create the conda environment, run `conda env create -f environment.yml`.
- To evaluate the attack systematically, download ImageNet validation dataset, AudioSet, and LLVIP and store in the folder `data/`. 
- Download [AudioClip checkpoint](https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt) in `bpe/`.
- Get the [PandaGPT](https://github.com/yxuansu/PandaGPT#2-running-pandagpt-demo-back-to-top) checkpoints in `PandaGPT/pretrained_ckpt` by following the instructions of `PandaGPT/pretrained_ckpt/README.md` 
- Get the [BindDiffusion](https://github.com/sail-sg/BindDiffusion) checkpoints in `PandaGPT/checkpoints` by following the instructions of `PandaGPT/README.md`


# Demonstration of Image Illusion on Text Generation
1. Run the `image_illusion_demo.ipynb` notebook.
2. Replace the existing image and aligned text with your own choices to generate an image illusion.
3. Run `text_generation_demo.ipynb` to see a quick demonstration of image illusions comprising text generation task.

# Experiment

## White Box

1. **Adjust Hyperparameters:**
   - Modify the corresponding configuration files to adjust the hyperparameters as needed.

2. **Run Experiments with Different Embedding Models:**

   - **Image Classification:**
     - `python adversarial_illusions.py imagenet/whitebox/imagebind`
     - `python adversarial_illusions.py imagenet/whitebox/openclip`
     - `python adversarial_illusions.py imagenet/whitebox/audioclip`

   - **Audio Classification:**
     - (Assigned to @Rishi)

   - **Thermal Image Classification:**
     - `thermal_illusion.ipynb` 

   - **Audio Retrieval:**
     - (Assigned to @Rishi)

## Black-box
  - **Run Transfer Attack Experiments:**
    - (Assigned to @Rishi)

  - **Run Query-based Attack Experiments:**
     - `python query_attack.py imagenet/query/imagebind`
     - `python query_attack.py imagenet/query/audioclip`
     - 
  - **Run Hybrid Attack Experiments:**
     - `python query_attack.py imagenet/hybrid/imagebind`
     - `python query_attack.py imagenet/hybrid/audioclip`

## Defense
  - **Certification:** `certification.ipynb`
  - **Anomaly Detection:** `anomaly_detection.ipynb`
  - **Feature Distillation:** `python adversarial_illusions_JPEG.py imagenet/whitebox/imagebind_jpeg`

Please feel free to email: [tz362@cornell.edu](mailto:tz362@cornell.edu) or raise an issue.


