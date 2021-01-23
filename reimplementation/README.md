### Reimplementation of Deep Single-Image Portrait Relighting

### Training
Use `train.py` along with the following command line arguments to run the training script.
```python3
usage: train.py [-h] [--epochs EPOCHS] [--high_resolution {True,False}] [--use_skip {True,False}] [--use_gan {True,False}] [--batch_size BATCH_SIZE] [--results_path RESULTS_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS, -e EPOCHS
                        Number of epochs to train the model
  --high_resolution {True,False}, -hr {True,False}
                        If fine-tuning on high-resolution data
  --use_skip {True,False}, -us {True,False}
                        Boolean flag to use skip connections
  --use_gan {True,False}, -ug {True,False}
                        Boolean flag to use gan loss
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        Batch size of the data
  --results_path RESULTS_PATH, -rp RESULTS_PATH
                        Path of the results
```
