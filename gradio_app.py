import gradio as gr
from fastai.vision.all import *
import skimage
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

learn = load_learner('cilantro.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Is it cilantro or parsley"
description = "Greens classifier for distinguishing cilantro and parsley and trained on images from DuckDuckGo."
outputs=gr.outputs.Label(num_top_classes=2)
examples = ['cilantro.jpg','parsley.jpg']
interpretation='default'
enable_queue=True

gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(400, 400)),
             outputs=outputs, examples = examples, title=title,
             description=description, interpretation=interpretation
             ).launch(share=True)