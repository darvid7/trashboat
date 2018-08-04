import json
from flask import Flask
app = Flask(__name__)

@app.route("/")
def main():
    return "Hello world"

# BE CAREFUL, MAKE SURE IT PRINTS "Loaded Cool ML Model before querying it".

# Read TensorFlow Graph.
with tf.gfile.FastGFile(os.path.join(cwd, "tf_files/retrained_graph.pb"), "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name="")

# Get softmax layer for predictions.
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

if __name__ == "__main__":

    app.run(threaded=True, port=1333)