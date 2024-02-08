name="threeview.waterfall.teddy.rabbit"
python generate.py --name ${name} --prompts "a waterfall" "a teddy bear" "a rabbit" --style "an oil painting of" --views identity rotate_cw rotate_ccw --num_samples 1 --num_inference_steps 30 --guidance_scale 10.0 --generate_1024 --save_dir results/test
# No animation implemented for this