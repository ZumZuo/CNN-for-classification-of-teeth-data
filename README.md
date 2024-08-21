This CNN aims to classify 256*256 images of teeth into seven main categories:
CaS: Calsium Sulfate, Typically used for teeth scafolding
CoS: Clinical Oral Surgery, Surgical interventions in the oral cavity, such as extractions, implant placements, or corrective jaw surgery.
Gum: Referring to the soft pink tissue surrounding teeth
MC: Metal Crown
OC: Oral Cavity
OLP: Oral Lichen Planus, A chronic inflammatory condition that affects the mucous membranes inside the mouth, leading to white patches, open sores, or swelling.
OT: Occlusal Therapy, Treatment aimed at correcting the bite or occlusion to improve the function and alignment of the teeth and jaws. This may involve the use of splints, braces, or other dental appliances.

The neural network's architecture invloves using 3 Conv layers, 2 Average and 1 Max Pooling layers, 2 Dropout layers and wrapping it all up with 2 Dense layers.
The ConvNet had image data compressed first to 192*192 before proceeding to train, mostly for dimensionality reduction, and was trained for 500 epochs on separate training and validation data, with final evaluation based on different test data, achieving 93% accuracy, with some run-to-run variance.
