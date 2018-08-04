import tensorflow as tf
import os

print("TESTING MODEL <3")

TEST_DATA_DIR = os.path.join(os.getcwd(), "tf_test_images")

CHILL_DIR = os.path.join(TEST_DATA_DIR, "chill")
VIOLENT_DIR = os.path.join(TEST_DATA_DIR, "violent")

chill_files = [os.path.join(CHILL_DIR, f) for f in os.listdir(CHILL_DIR) if f.endswith('.jpg')]
violent_files = [os.path.join(VIOLENT_DIR, f) for f in os.listdir(VIOLENT_DIR) if f.endswith('.jpg')]

print("chill %s" % len(chill_files))
print("violent %s" % len(violent_files))

test_data = []
test_data.extend(chill_files)

mapper = {
    "chill": 0,
    "violent": 1
}
rev_mapper = {
    0: "chill",
    1: "violent"
}

test_labels = [mapper["chill"] for _ in range(len(chill_files))]
test_data.extend(violent_files)

test_labels.extend([mapper["violent"] for _ in range(len(violent_files))])

print(test_labels.count(0))
print(test_labels.count(1))

# TESTING FAM <3

cwd = os.getcwd()

label_lines = [line.rstrip() for line in tf.gfile.GFile(os.path.join(cwd, "tf_files/retrained_labels.txt"))]

with tf.gfile.FastGFile(os.path.join(cwd, "tf_files/retrained_graph.pb"), "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name="")

predictions = []

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    # Evaluate against my test set!
    for i in range(len(test_labels)):
        image_path = test_data[i]
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        prediction = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        # print(prediction)
        top_k = prediction[0].argsort()[-len(prediction[0]):][::-1]
        top_preds = []
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = prediction[0][node_id]
            top_preds.append((human_string, score))
        predictions.append(top_preds)
        if i % 10 == 0:
            print("label: %s, guess %s (score = %.5f)" % (rev_mapper[test_labels[i]], human_string, score))

correct = 0

for i, prediction_tuple in enumerate(predictions):
    best_guess = prediction_tuple[0]
    human_string, score = best_guess
    actual_label = test_labels[i]
    if mapper[human_string] == actual_label:
        correct += 1
    print("label: %s, guess %s (score = %.5f)" % (rev_mapper[test_labels[i]], human_string, score))
print("\nevaluation\n")
print(correct)
print(len(test_data))
print("accuracy %s" % (correct/len(test_data)))