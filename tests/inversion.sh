name="negate.landscape.houseplants"
python generate.py --name ${name} --prompts "a landscape" "houseplants" --style "a lithograph of" --views identity negate --num_samples 1 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024 --save_dir results/test
python animate.py --im_path results/test/${name}/0000/sample_64.png --metadata_path results/test/${name}/metadata.pkl
python animate.py --im_path results/test/${name}/0000/sample_256.png --metadata_path results/test/${name}/metadata.pkl
python animate.py --im_path results/test/${name}/0000/sample_1024.png --metadata_path results/test/${name}/metadata.pkl