import io
import zipfile

import tensorflow as tf
from PIL import Image
from visual_geolocation.utils import geocell_to_class, coord_to_geocell




def load_image_from_zip(IMG_FOLDER, prefix, img_id):
    """Load a single image from a local zip archive without extracting it to disk.

    Args:
        IMG_FOLDER: path to the local zip file (e.g. "00.zip")
        prefix: folder name inside the zip matching the zip file (e.g. "00")
        img_id: numeric id of the image, matching the filename without extension

    Returns:
        A PIL Image object.
    """
    with zipfile.ZipFile(IMG_FOLDER) as z:
        with z.open(f"{prefix}/{img_id}.jpg") as f:
            img_bytes = f.read()
    return Image.open(io.BytesIO(img_bytes))


def build_labeled_dataframe(df, IMG_FOLDER):
    """Filter a dataframe to keep only rows whose image is present in the given zip,
    and compute the geocell label for each row from its coordinates.

    Args:
        df: source dataframe with at least 'id', 'latitude', 'longitude' columns
        IMG_FOLDER: path to the local zip file
        coord_to_geocell: function that maps (lon, lat) to a geocell id

    Returns:
        A filtered dataframe with an added 'geocell' column.
    """
    with zipfile.ZipFile(IMG_FOLDER) as z:
        zip_ids = [
            int(n.split('/')[-1].replace('.jpg', ''))
            for n in z.namelist() if n.endswith('.jpg')
        ]

    subset = df[df['id'].isin(zip_ids)][['id', 'latitude', 'longitude', 'unique_country']].copy()
    subset['geocell'] = subset.apply(
        lambda row: coord_to_geocell(row['longitude'], row['latitude']),
        axis=1
    )
    subset['class'] = subset['geocell'].apply(geocell_to_class)
    return subset


def load_image_and_label(img_id, label, IMG_FOLDER, prefix, img_size):
    """Python-level loader called through tf.py_function.
    Converts a tensor image id into a resized numpy image array paired with its label.

    Args:
        img_id: tensor holding the image id
        label: tensor holding the label
        IMG_FOLDER: path to the local zip file (fixed for this dataset instance)
        prefix: folder name inside the zip (fixed for this dataset instance)
        img_size: target (height, width) for resizing

    Returns:
        Tuple (image_array, label) as float32 tensors.
    """
    img_id = img_id.numpy()
    img = load_image_from_zip(IMG_FOLDER, prefix, img_id)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.image.resize(img_array, img_size)
    return img_array, label


def make_tf_dataset(df, IMG_FOLDER, img_size=(64, 64), batch_size=16):
    """Build a tf.data.Dataset that loads images on the fly from a local zip archive.

    Args:
        df: dataframe with 'id' and 'geocell' columns (output of build_labeled_dataframe)
        IMG_FOLDER: path to the local zip file
        img_size: target (height, width) for resizing every image
        batch_size: number of samples per batch

    Returns:
        A batched tf.data.Dataset yielding (image, label) pairs.
    """
    prefix = IMG_FOLDER.replace(".zip", "")

    ids = df['id'].tolist()
    labels = df['class'].tolist()


    def wrapper_tf(img_id, label):
        """TensorFlow-graph-compatible wrapper around load_image_and_label.
        Bridges tf.data's graph execution with the plain Python image-loading logic
        via tf.py_function, then restores the static shapes lost in that call.

        Args:
            img_id: tensor holding the image id
            label: tensor holding the label

        Returns:
            Tuple (image, label) as float32 tensors with shapes explicitly set.
        """
        img, label = tf.py_function(
            func=lambda i, l: load_image_and_label(i, l, IMG_FOLDER, prefix, img_size),
            inp=[img_id, label],
            Tout=[tf.float32, tf.float32]
        )
        img.set_shape([img_size[0], img_size[1], 3])
        label.set_shape([])
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((ids, labels))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(wrapper_tf)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def preprocess_features():


    # Color encoded on 8 bits -> 255
    # X_train_shaped = X_train /255
    # X_test_shaped = X_test /255


    pass

    #retunr X_preproc
