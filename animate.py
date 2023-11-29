from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

import torchvision.transforms.functional as TF

from visual_anagrams.views import get_views
from visual_anagrams.utils import get_courier_font_path


def draw_text(image, text, fill=(0,0,0), frame_size=384, im_size=256):
    image = image.copy()

    # Font info
    font_path = get_courier_font_path()
    font_size = 16

    # Make PIL objects
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    
    # Center text horizontally, and vertically between
    # illusion bottom and frame bottom
    text_position = (0, 0)
    bbox = draw.textbbox(text_position, text, font=font, align='center')
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_left = (frame_size - text_width) // 2
    text_top = int(3/4 * frame_size + 1/4 * im_size - 1/2 * text_height)
    text_position = (text_left, text_top)

    # Draw text on image
    draw.text(text_position, text, font=font, fill=fill, align='center')
    return image


def easeInOutQuint(x):
    # From Matthew Tancik: 
    # https://github.com/tancik/Illusion-Diffusion/blob/main/IllusionDiffusion.ipynb
    if x < 0.5:
        return 4 * x**3
    else:
        return 1 - (-2 * x + 2)**3 / 2


def animate_two_view(
        im_path,
        view,
        prompt_1,
        prompt_2,
        save_video_path='tmp.mp4',
        hold_duration=120,
        text_fade_duration=10,
        transition_duration=60,
        im_size=256,
        frame_size=384,
):
    '''
    TODO: Assuming two views, first one is identity
    '''
    im = Image.open(im_path)

    # Make list of frames
    frames = []

    # Make frames for two views 
    frame_1 = view.make_frame(im, 0.0)
    frame_2 = view.make_frame(im, 1.0)

    # Display frame 1 with text
    frame_1_text = draw_text(frame_1, 
                             prompt_1, 
                             frame_size=frame_size, 
                             im_size=im_size)
    frames += [frame_1_text] * (hold_duration // 2)

    # Fade out text 1
    for t in np.linspace(0,1,text_fade_duration):
        c = int(t * 255)
        fill = (c,c,c)
        frame = draw_text(frame_1, 
                          prompt_1, 
                          fill=fill,
                          frame_size=frame_size, 
                          im_size=im_size)
        frames.append(frame)

    # Transition view 1 -> view 2
    for t in tqdm(np.linspace(0,1,transition_duration)):
        t_ease = easeInOutQuint(t)
        frames.append(view.make_frame(im, t_ease))

    # Fade in text 2
    for t in np.linspace(1,0,text_fade_duration):
        c = int(t * 255)
        fill = (c,c,c)
        frame = draw_text(frame_2,
                          prompt_2,
                          fill=fill,
                          frame_size=frame_size, 
                          im_size=im_size)
        frames.append(frame)

    # Display frame 2 with text
    frame_2_text = draw_text(frame_2, 
                             prompt_2, 
                             frame_size=frame_size, 
                             im_size=im_size)
    frames += [frame_2_text] * (hold_duration // 2)

    # "Boomerang" the clip, so we get back to view 1
    frames = frames + frames[::-1]

    # Move last bit of clip to front
    frames = frames[-hold_duration//2:] + frames[:-hold_duration//2]

    # Convert PIL images to numpy arrays
    image_array = [imageio.core.asarray(frame) for frame in frames]

    # Save as video
    print('Making video...')
    imageio.mimsave(save_video_path, image_array, fps=30)



if __name__ == '__main__':
    import argparse
    import pickle
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--im_path", required=True, type=str, help='Path to the illusion to animate')
    parser.add_argument("--save_video_path", default=None, type=str, 
        help='Path to save video to. If None, defaults to `im_path`, with extension `.mp4`')
    parser.add_argument("--metadata_path", default=None, type=str, help='Path to metadata. If specified, overrides `view` and `prompt` args')
    parser.add_argument("--view", default=None, type=str, help='Name of view to use')
    parser.add_argument("--prompt_1", default='', nargs='+', type=str,
        help='Prompt for first view. Passing multiple will join them with newlines.')
    parser.add_argument("--prompt_2", default='', nargs='+', type=str,
        help='Prompt for first view. Passing multiple will join them with newlines.')
    args = parser.parse_args()


    # Load image
    im_path = Path(args.im_path)

    # Get save dir
    if args.save_video_path is None:
        save_video_path = im_path.with_suffix('.mp4')

    if args.metadata_path is None:
        # Join prompts with newlines
        prompt_1 = '\n'.join(args.prompt_1)
        prompt_2 = '\n'.join(args.prompt_2)

        # Get paths and views
        view = get_views([args.view])[0]
    else:
        with open(args.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        view = metadata['views'][1]
        m_args = metadata['args']
        prompt_1 = f'{m_args.style} {m_args.prompts[0]}'.strip()
        prompt_2 = f'{m_args.style} {m_args.prompts[1]}'.strip()


    # Animate
    animate_two_view(
            im_path,
            view,
            prompt_1,
            prompt_2,
            save_video_path=save_video_path,
            hold_duration=120,
            text_fade_duration=10,
            transition_duration=45,
            im_size=256,
            frame_size=384,
        )