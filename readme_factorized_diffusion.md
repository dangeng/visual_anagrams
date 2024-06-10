# Factorized Diffusion: Perceptual Illusions by Noise Decomposition

[Daniel Geng*](https://dangeng.github.io/), [Aaron Park*](https://inbumpark.github.io/), [Andrew Owens](https://andrewowens.com/)

## [[Arxiv](https://arxiv.org/abs/2404.11615)] [[Website](https://dangeng.github.io/factorized_diffusion/)] [[Colab (Free Tier)](https://colab.research.google.com/drive/1S9v0m9fgAw4MDdsLVHAdaRx94L4dwTPw?usp=sharing)]

[![Open In Colab (Free Tier)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S9v0m9fgAw4MDdsLVHAdaRx94L4dwTPw?usp=sharing) <sub>(Free Tier)</sub>

This readme includes instructions on how to use Factorized Diffusion, which allows conditioning different components of an image on different text prompts. For example, conditioning low frequencies of an image on one prompt but high frequencies on another results in [hybrid images](https://stanford.edu/class/ee367/reading/OlivaTorralb_Hybrid_Siggraph06.pdf) ([wikipedia](https://en.wikipedia.org/wiki/Hybrid_image)). We present a number of different decompositions that can be used with Factorized Diffusion below. Please read our paper or visit our website for more details.

## Installation

Please see the main [`readme.md`](https://github.com/dangeng/visual_anagrams) for installation instructions.

## Usage and Important Arguments

As a basic example, the following command can be run to create hybrid images, and saves them to `./results/hybrid/`:

```
python generate.py --save_dir results --name hybrid --prompts "a painting of a panda" "a painting of a flower arrangement" --views low_pass high_pass --num_samples 8 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024 --reduction sum
```

The results can then be animated by running the following command:

```
python animate.py --im_path results/hybrid/0000/sample_1024.png --metadata_path results/hybrid/metadata.pkl
```

Below we describe the more important arguments to the `generate.py` script:

- `--prompts`: A list of prompts for illusions. Should match number of `--views`.
- `--views`: A list of views to use. These specify what components of an image are being conditioned. Must match the number of `--prompts`. For a list of views see the `get_views` function in `visual_anagrams/views/__init__.py`, or see the examples below.
- `--reduction`: **It is important that this is set to `sum`,** as opposed to `mean` as in Visual Anagrams. This is because we define our decompositions to be a sum of components.
- `--view_args`: Used to pass arguments to the views. For example, this can be used to specify the strength of the filters for the hybrid images, or what `scales` to use for the scaling decomposition (see below). Must match number of `--views` and `--prompts`.

## Hybrid Images

The following command generates hybrid images:

```
python generate.py --save_dir results --name hybrid --prompts "a painting of a panda" "a painting of a flower arrangement" --views low_pass high_pass --num_samples 8 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024 --reduction sum
```

It uses the `"low_pass"` and `"high_pass"` views to condition low and high frequencies on different prompts. This results in an image that changes appearance depending on the distance it is viewed from (or if you squint). The default filters use Gaussian blurring with a `sigma` value of `2.0`. The strength of these filters can be adjusted by passing `sigma` values using the `--view_args` flag. For example, adding `--view_args 3.0 3.0` to the command uses a `sigma` value of 3.0. A lower `sigma` makes the low frequency prompt easier to see. A higher `sigma` makes the high frequency prompt easier to see.

## Color Hybrids

The following command generates color hybrids:

```
python generate.py --save_dir results --name color --prompts "landscape, oil painting style" "tiger, oil painting style" --views grayscale color --num_samples 8 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024 --seed 0 --reduction sum
```

It generates images that change appearance when colorized or turned grayscale. Because humans don't see color well under dim light, these images change appearance when taken from a dark room to a bright one. These images are made by conditioning the grayscale component on one prompt and the color component on another, by using the `"grayscale"` and `"color"` views. 

## Motion Hybrids

The following command generates motion hybrids:

```
python generate.py --save_dir results --name motion --prompts "a photo of a panda" "a photo of a canyon" --views motion motion_res --num_samples 8 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024 --seed 0 --reduction sum
```

Motion hybrids change appearance under motion blur. This is done by conditioning a motion blurred version of the image with one prompt, and the "residual" on a different prompt, and can be thought of as a "directional" hybrid image. The corresponding views are specified by `"motion"` and `"motion_res"`.

## Triple Hybrids

The following command generates triple hybrids:

```
python generate.py --save_dir results --name triple_hybrid --prompts "a lithograph of a yin yang" "a lithograph of a skull" "a lithograph of waterfalls" --views triple_low_pass triple_medium_pass triple_high_pass --num_samples 1 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024 --seed 7 --reduction sum
```

Triple hybrids are hybrid images, but with three different things in them. This is done by conditioning low, medium, and high frequency components of the image on different prompts by applying appropriate bandpasses. **Note, triple hybrids are quite hard to generate**, given how many things we are trying to put in a single image. The above command uses arguments that we found worked well (e.g. `--seed 7` and specific `sigma` values). Be aware that finding good triple hybrids may take a good amount of sifting through images.

## Scaling Decomposition

For the a scaling decomposition (see [paper](https://arxiv.org/abs/2404.11615) or [website](https://dangeng.github.io/factorized_diffusion/)), our method reduces exactly to various prior methods, depending on the choice of constants. **Note:** for these scaling decompositions, the `--guidance_scale` parameter must be set to `1.0`, otherwise the scaling will be incorrect.

**Classifier Free Guidance** ([Ho and Salimans](https://arxiv.org/abs/2207.12598)): To recover classifier free guidance with our framework, run the following command:

```
generate.py --save_dir results --name scale_negate_no_male_simple --prompts "" "a corgi" --views scale scale --num_samples 8 --num_inference_steps 30 --guidance_scale 1.0 --generate_1024  --seed 0 --reduction sum --view_args -6.5 7.5
```

The above uses CFG with a scale of `7.5`, and uses the fact that the model is trained so that conditioning on the empty string `""` gives an unconditional estimate. The results can be compared to standard conditional sampling:

```
generate.py --save_dir results --name scale_negate_no_male_simple --prompts "a corgi" --views scale --num_samples 8 --num_inference_steps 30 --guidance_scale 1.0 --generate_1024 --seed 0 --reduction sum --view_args 1.0 
```

**Compositionality:** Introduced by [Liu et al.](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/), compositionality allows us to condition on multiple text prompts simultaneously. To test this out, run the following command, which composes the text `"a photo of a church"` and `"a photo of a purple sky"` together:

```
python generate.py --save_dir results --name scale --prompts "a photo of a church" "a photo of a purple sky" --views scale scale --num_samples 8 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024 --seed 0 --reduction sum --view_args 0.5 0.5
```

Note: while we could technically just condition on `"a photo of a church and a purple sky"`, compositionality forms the basis of other quite interesting techniques, such as [Visual Anagrams](https://dangeng.github.io/visual_anagrams/) and [Images that Sound](https://ificl.github.io/images-that-sound/), in which the compositions are done over different transformations of an image, or different modalities. In addition, at the time of publication of the Liu et al. papers, diffusion models were not very good when conditioned on these conjunctions, so this result was quite interesting.

**Negation:** The scaling decomposition also recovers various ways to do negation, such as a technique also proposed by [Liu et al.](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/), and [negative prompting](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Negative-prompt). To use Liu et al.'s method, run the following command:

```
generate.py --save_dir results --name scale_negate_no_male_simple --prompts "" "a photo of a person" "female" --views scale scale scale --num_samples 8 --num_inference_steps 30 --guidance_scale 1.0 --generate_1024 --seed 0 --reduction sum --view_args 1.0 15.0 -15.0
```

and to use negative prompting run the following command:

```
generate.py --save_dir results --name scale_negate_no_male_simple --prompts "a photo of a person" "female" --views scale scale --num_samples 8 --num_inference_steps 30 --guidance_scale 1.0 --generate_1024 --seed 0 --reduction sum --view_args 15.0 -14.0
```

## Inverse Problems

A special case of our method is if we fix one component (say, extracted from some reference image) while conditioning on another component. This gives us a way to solve inverse problems with a diffusion model, and has been studied in numerous works [[1]](https://arxiv.org/abs/2011.13456) [[2]](https://arxiv.org/abs/2112.05146) [[3]](https://arxiv.org/abs/2201.11793) [[4]](https://arxiv.org/abs/2201.09865) [[5]](https://arxiv.org/abs/2212.00490) [[6]](https://arxiv.org/abs/2108.02938) [[7]](https://arxiv.org/abs/2206.02779). Our contribution is (1) to show that our more general method reduces to (approximately) these approaches, and (2) to use it to generate hybrid images from existing images, which the following command does:

```
generate.py --save_dir results --name inverse --prompts "" "a lithograph of waterfalls" --views low_pass high_pass --num_samples 8 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024 --seed 0 --reduction sum --ref_im_path ./assets/einstein.png
```

Here, the `--ref_im_path` specifies where the reference image path. We use the first component (here, the `"low_pass"`) of the reference image, and generate the second component (here, the high frequencies). This is done by projecting after each denoising step.

As another example, we can colorize grayscale images (which has thoroughly been explored in prior work) by running the following command:

```
generate.py --save_dir results --name inverse_color --prompts "" "a photo of birds" --views grayscale color --num_samples 8 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024 --seed 0 --reduction sum --ref_im_path ./assets/birds.png
```

Here, the grayscale component comes from the `birds.png` image while the color component is generated conditioned on the text `"a photo of birds"`.


## Animating

The below decompositions should have animations implemented for them, and can be generated by using `animate.py`:

- Hybrids
- Color Hybrids
- Motion Hybrids (note, under the hood this uses a custom function called `animate_two_view_motion_blur`)
- Inverse Hybrids