import os
import random
from shutil import copyfile
from shutil import rmtree

# NOTE: Fix paths if you use this.

print(os.getcwd())

VIOLENT_DIR = "./custom_classifier/data_4_model/violent"
CHILL_DIR = "./custom_classifier/data_4_model/chill"
OUTPUT_DIR = "./custom_classifier/tf_files/data"
OUTPUT_TEST_DIR = "./custom_classifier/tf_test_files"

violent_images = [f for f in os.listdir(VIOLENT_DIR) if f.endswith('.jpg')]
chill_images = [f for f in os.listdir(CHILL_DIR) if f.endswith('.jpg')]

print("violent_images: %s" % len(violent_images))
print("chill_images: %s" % len(chill_images))

# 80:20 train test split.

num_test_violent = int(len(violent_images) * 0.2)
num_test_chill = int(len(chill_images) * 0.2)

print("violent test images %s" % num_test_violent)
print("chill test images %s" % num_test_chill)

chosen_test_violent = set(random.sample(range(0, len(violent_images)), num_test_violent))
chosen_test_chill = set(random.sample(range(0, len(violent_images)), num_test_chill))

train_violent_images = []
test_violent_images = []

for i in range(len(violent_images)):
    if i in chosen_test_violent:
        test_violent_images.append(violent_images[i])
    else:
        train_violent_images.append(violent_images[i])

train_chill_images = []
test_chill_images = []

for j in range(len(chill_images)):
    if j in chosen_test_chill:
        test_chill_images.append(chill_images[j])
    else:
        train_chill_images.append(chill_images[j])

# Copy files to dirs.
if os.path.exists(OUTPUT_DIR):
    rmtree(OUTPUT_DIR)
if os.path.exists(OUTPUT_TEST_DIR):
    rmtree(OUTPUT_TEST_DIR)

os.makedirs(OUTPUT_DIR)
os.makedirs(OUTPUT_TEST_DIR)

os.makedirs(os.path.join(
    OUTPUT_DIR, "train", "chill"
))
os.makedirs(os.path.join(
    OUTPUT_DIR, "test", "chill"
))

os.makedirs(os.path.join(
    OUTPUT_DIR, "train", "violent"
))
os.makedirs(os.path.join(
    OUTPUT_DIR, "test", "violent"
))

for file in train_chill_images:
    copyfile(
        os.path.join(
            CHILL_DIR,
            file),
        os.path.join(
            OUTPUT_DIR,
            "chill",
            file)
    )

for file in test_chill_images:
    copyfile(
        os.path.join(
            CHILL_DIR,
            file),
        os.path.join(
            OUTPUT_TEST_DIR,
            "chill",
            file)
    )

for file in train_violent_images:
    copyfile(
        os.path.join(
            VIOLENT_DIR,
            file),
        os.path.join(
            OUTPUT_DIR,
            "violent",
            file)
    )


for file in test_violent_images:
    copyfile(
        os.path.join(
            VIOLENT_DIR,
            file),
        os.path.join(
            OUTPUT_TEST_DIR,
            "violent",
            file)
    )
