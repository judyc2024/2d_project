This repository contains code to separate subcaption and subfigures using the open pmc repository, 
then to use the OpenAI API key to do ocr detection on subfigures. And then finally, to use a custom 
matching script in order to match letter-tagged subfigure and subcaption. 

The workflow goes as the following: 
- First, we use the "subcaption_brain_ct_1_20.py" to separation subcaptions, and use "subfigure_brain_ct_1_20.py" to separate subfigures. Make sure add --run at the end of your terminal command. 
- Then, we use the "analyze_image_base64.py" script to detect the letters in the subfigures 
- Then, use "match_figure_caption.py" to match subfigure and subcaption 

I hope this helps navigate the workflow! 

- by Judy Chung 