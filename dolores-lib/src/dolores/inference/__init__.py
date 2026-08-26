"""Load a serialized model and score payloads against it.

The entire import surface of the inference server. Deliberately separate from
:mod:`dolores.experiment`: a different consumer with a different lifecycle, and
keeping it standalone is what lets the server's import stay one name wide.

Never talks to MLflow. It reads a portable artifact and predicts.
"""
