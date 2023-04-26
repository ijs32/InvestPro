# InvestPro

This project aims to teach a ML model to read company 10K and 10Q statements in order to predict company performance.

I have begun InvestPro as a simple 1D CNN but this is clearly the wrong approach. I will now work towards building a model on the Transformer architecture.

WIP

## Project Structure

Everything of importance is found within the ./src directory.

within `./src/dataset.py` you will find the dataset class.

within `./src/execute.py` you will find the interface for the `Trainer` class found in `trainer.py`.

within `./src/loss.py` you will find my simple custom loss function for the model. seeing as the outputs and labels are always decimal values, I created a 
MeanSqrtError loss function that effectively does the same job as MSE but for decimal values. IE. a difference between prediction and label of `0.25` becomes `0.5` rather than `0.0625`. this in turn punishes predictions further from the correct value.

`model.py` contains the structure of the model.

finally `trainer.py` contains all the code involved with the actual training of the model.

I have to withhold all of the data due to the sheer size of it. I unfortunately cannot upload 8 GBs of training data. in the future, if I can find the time, I will consider writing a getting started section explaining how and where you can get the data. for the time being, you can try to read through the absolute mess that is `./src/utils/scripts.py` to piece together how I did it.