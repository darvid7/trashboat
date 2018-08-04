import json
from flask import Flask
from flask import request
import tensorflow as tf
import os

cwd = os.getcwd()

label_lines = [line.rstrip() for line in tf.gfile.GFile(os.path.join(cwd, "tf_files/retrained_labels.txt"))]

# Load TensorFlow Graph.
with tf.gfile.FastGFile(os.path.join(cwd, "tf_files/retrained_graph.pb"), "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name="")
# Get softmax layer for predictions.

# Make session and keep it open.
sess = tf.Session()
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

print("tensorflow stuff loaded")
# Do flask things.
app = Flask(__name__)

@app.route("/")
def main():
    return "Hello world"

@app.route("/classify", methods=["POST"])
def classify():
    # Expects <class 'bytes'>
    image_as_bytes = request.data
    print(image_as_bytes.__class__)

    # Expects image data as bytes i think.
    image_data = image_as_bytes
    print(image_data.__class__)
    
    #tf.gfile.FastGFile(image_path, 'rb').read()
    prediction = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

    # Get prediction.
    top_k = prediction[0].argsort()[-len(prediction[0]):][::-1]
    outcome = []
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = prediction[0][node_id]
        outcome.append((human_string, float(score)))
     # returns something like [["chill", 0.9], ["violent", 0.1]]
    return json.dumps({"classifications": outcome})

if __name__ == "__main__":

    app.run(threaded=True, port=1333)