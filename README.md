This week's data is the same as last week's satellite images, only difference is I used a pretrained U-Net with ResNet34 backbone for my image segmentation model to label aquatic areas, finetuning it for better fitting to the label masks and better predictions.
In case you want to deploy the model yourself using Flask, define the model architecture and then use load_state_dict, then structure your files as follows:
/your_flask_app
│
├── /static
│   ├── /css
│   │   └── style.css
│   └── /js
│       └── script.js
├── /templates
│   └── index.html
├── app.py
└── week4_task.pth
