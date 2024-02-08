name="flip.campfire.man"
python generate.py --name ${name} --prompts "an oil painting of people around a campfire" "an oil painting of an old man" --views identity flip --num_samples 1 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024 --save_dir results/test
python animate.py --im_path results/test/${name}/0000/sample_64.png --metadata_path results/test/${name}/metadata.pkl
python animate.py --im_path results/test/${name}/0000/sample_256.png --metadata_path results/test/${name}/metadata.pkl
python animate.py --im_path results/test/${name}/0000/sample_1024.png --metadata_path results/test/${name}/metadata.pkl