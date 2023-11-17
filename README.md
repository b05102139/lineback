# lineback: PyTorch Lightning Callback for LINE
This package contains a simple callback for PyTorch Lightning models, which sends progress reports through LINE. 

## Installation
To install the dependencies, do this:
```python
pip install -r requirements.txt
```

And for the package itself (assuming you are within the repo):
```python
pip install .
```

## Usage
The callback can be used as follows, where the only necessary params are the LINE Notify API token that you should obtain from their site (https://notify-bot.line.me/en/), and a name for the process:
```python
from lineback import lineback
API_TOKEN=""
PROCESS_NAME="autoencoder"
# insert relevant pytorch lightning code here
trainer = Trainer(callbacks=[lineback(token=API_TOKEN, process_name=PROCESS_NAME)])
```

Your model will then periodically send you progress reports of all logged metrics after each epoch, where the reports come in the form of concrete numbers and plots.

## Notes
This is still very barebones, so any pull requests are welcome. 