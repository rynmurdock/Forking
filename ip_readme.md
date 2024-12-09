# Image to Video Prior

Just an ip adapter on LTX. Generalizes to video.


- timestep selection? 
- cache good and bad pos & neg prompt for all training?
- More varied data
- Masked loss so only first frame/two matters, not caring re padding/replicated frames (check how VAE functions!!!)
- Put on an a100 on Colab
- Negative & positive prompts to get more movement