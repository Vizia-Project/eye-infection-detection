import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import tensorflowjs as tfjs

def printHelp(): 
    print(
        """
        Usage: python3 model-converter.py model1.keras [model2.keras ...] --target-dir /path/to/target/
        """
    )

if __name__ == "__main__":
    if '--target-dir' not in sys.argv:
        print("Missing argument: --target-dir")
        printHelp()
        exit(1)

    model_files = sys.argv[1:-2]
    target_dir = sys.argv[-1]
    
    print("Files: ", model_files)
    print("Target dir: ", target_dir)

    for model_file in model_files:
        if not model_file.endswith(".keras"):
            print(f"{model_file} is not a keras model or not in keras format!")
            continue

        model = tf.keras.models.load_model(model_file)
        tfjs.converters.save_keras_model(model, target_dir)

        print(f"Successfully converted {model_file} to TensorflowJS format!")
