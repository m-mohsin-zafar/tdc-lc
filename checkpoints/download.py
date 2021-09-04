import gdown
import subprocess


_idClassifier = '1jm3rakiXrO1kNiIoyVibQHMlW6QEWMD7'
_idDetector = '1TQAMCxjv1ieWpZOUftNazerQD76iqtKZ'
_urlForClassifierCkpt = "https://drive.google.com/uc?id={id}".format(id=_idClassifier)
_urlForDetectorCkpt = "https://drive.google.com/uc?id={id}".format(id=_idDetector)

try:
    gdown.download(_urlForClassifierCkpt)
    gdown.download(_urlForDetectorCkpt)
except Exception as exc:
    print('Something went wrong with Downloading the Checkpoints')
    print(exc)
