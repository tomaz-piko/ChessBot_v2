import argparse
from configs import trainingConfig
import os
import json
import numpy as np
from model import update_trt_model
from strength_test import do_strength_test
import tensorflow as tf
from multiprocessing import Process
from datetime import datetime
from functools import partial


def load_training_info(config):
    """ Load the training information from disk. If no such file exists yet, return a default training_info.
    Default training_info is:
    {
        'current_epoch': 0,
    }

    Args:
        config (dict): A dictionary containing the configuration for the project.

    Returns:
        dict: A dictionary to save any training information needed on subsequent runs.
    """
    training_info_path = os.path.join(config['project_dir'], 'data', 'training_info.json')
    if os.path.exists(training_info_path):
        with open(training_info_path, 'r') as f:
            training_info = json.load(f)
            return training_info
    else:
        return {
            'current_epoch': 0,
        }


def save_training_info(config, training_info):
    """ Save the training information to disk. Create a new file or overwrite an existing one.

    Args:
        config (dict): A dictionary containing the configuration for the project.
        training_info (dict): A dictionary to save any training information needed on subsequent runs.
    """
    training_info_path = os.path.join(config['project_dir'], 'data', 'training_info.json')
    with open(training_info_path, 'w') as f:
        json.dump(training_info, f)


def count_available_samples(config):
    """ Count the number of available samples for training.

    Args:
        config (dict): A dictionary containing the configuration for the project.

    Returns:
        int: The number of available samples for training.
        int: The number of samples to use after sampling.
    """
    sample_dir = os.path.join(config['project_dir'], 'data', 'selfplay_data')
    total_samples = 0
    sampling_ratio = config['sampling_ratio']

    for filename in os.listdir(sample_dir):
        if filename.endswith('.npz'):
            num_positions = int(filename.split('_')[0])
            total_samples += num_positions

    return total_samples, int(total_samples * sampling_ratio)


def load_samples(config, load: int, take: int):
    """ Load the samples for training.

    Args:
        config (dict): A dictionary containing the configuration for the project.
        load (int): The number of samples to load.
        take (int): The number of samples to take from the loaded samples. (Randomly selected, the rest will be discarded)

    Returns:
        list: An array containing the samples for training. Each sample is a tuple of (images, (search_stats, terminal_values)).
    """
    print(f"Loading {load} samples from {os.path.join(config['project_dir'], 'data', 'selfplay_data')}")
    print(f"Taking {take} samples from the loaded samples.")
    print(f"Sampling ratio: {config['sampling_ratio']}")
    rng = np.random.default_rng()

    sample_dir = os.path.join(config['project_dir'], 'data', 'selfplay_data')
    restore_dir = os.path.join(config['restore_dir'])
    samples = []

    # Get list of files and sort them by the number of samples (least samples first)
    files = [f for f in os.listdir(sample_dir) if f.endswith('.npz')]
    rng.shuffle(files)

    for filename in files:
        file_path = os.path.join(sample_dir, filename)
        try:           
            data = np.load(file_path)
            images = data['images']
            search_stats = data['search_stats']
            terminal_values = data['terminal_values']
            num_file_samples = len(images)

            if len(samples) + num_file_samples <= load:
                for image, search_stat, terminal_value in zip(images, search_stats, terminal_values):
                    samples.append((image, (search_stat, terminal_value)))
                data.close()
                del images, search_stats, terminal_values
                os.rename(file_path, os.path.join(restore_dir, filename))           
            else:
                remaining_samples = load - len(samples)
                for image, search_stat, terminal_value in zip(images[:remaining_samples], search_stats[:remaining_samples], terminal_values[:remaining_samples]):
                    samples.append((image, (search_stat, terminal_value)))
                data.close()                
                # Save the remaining data back to a new file with a new name, {remaining_samples}_{timestamp}
                timestamp = datetime.now().strftime('%F_%T.%f')[:-3]
                new_filename = f"{remaining_samples}_{timestamp}.npz"
                np.savez(os.path.join(sample_dir, new_filename), 
                            images=images[remaining_samples:], 
                            search_stats=search_stats[remaining_samples:],
                            terminal_values=terminal_values[remaining_samples:])
                del images, search_stats, terminal_values
                # Delete the old file
                os.remove(file_path)
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            # Move the file to the restore directory
            os.remove(file_path)
        if len(samples) >= load:
                break
    del files
    
    # Randomly select num_selected_samples from samples, each sample can be selected only once
    indices = rng.choice(len(samples), size=take, replace=False)
    samples = [samples[i] for i in indices]
    rng.shuffle(samples)
    return samples


def create_tf_record(config, samples):
    timestamp = datetime.now().strftime('%F_%T.%f')[:-3]
    save_path = os.path.join(config['project_dir'], 'data', 'train_data', f"{timestamp}.tfrecords")
    with tf.io.TFRecordWriter(save_path) as writer:
        for sample in samples:
            image, (search_stats, terminal_value) = sample
            
            image_features = tf.train.Feature(int64_list=tf.train.Int64List(value=image))
            search_stats_features = tf.train.Feature(float_list=tf.train.FloatList(value=search_stats))
            terminal_value_features = tf.train.Feature(float_list=tf.train.FloatList(value=[terminal_value]))

            example = tf.train.Example(features=tf.train.Features(feature={
                "image": image_features,
                "value_head": terminal_value_features,
                "policy_head": search_stats_features
            }))
            writer.write(example.SerializeToString())


def do_sts_test(config, current_epoch, model_version="latest"):
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

    time_limit = config.get('sts_time_limit', 1.0)
    num_agents = config.get('sts_num_actors', 6)
    sts_rating = do_strength_test(time_limit=time_limit, num_agents=num_agents, model_version=model_version)
    sts_summary_writter = tf.summary.create_file_writer(os.path.join(config['project_dir'], 'data', 'logs', 'sts'))
    with sts_summary_writter.as_default():
        tf.summary.scalar("ELO Rating", sts_rating, step=current_epoch // config['sts_test_interval'])


def update_trt_model_process(config, model_version="latest", precision_mode="FP16", build_model=True):
    """ Update the TensorRT model in a separate process.
    """
    p = Process(target=update_trt_model, args=(config, model_version, precision_mode, build_model))
    p.start()
    p.join()


def train_model(config, training_info, batch_size, epochs):
    """Train the model using the provided samples.

    Args:
        config (dict): A dictionary containing the configuration for the project.
        training_info (dict): A dictionary to save any training information needed on subsequent runs.
        samples (list): An array containing the samples for training. Each sample is a tuple of ((images, search_stats), terminal_values).
    """
    def read_tfrecord(example_proto):
        feature_description = {
            "image": tf.io.FixedLenFeature([109], tf.int64),
            "value_head": tf.io.FixedLenFeature([1], tf.float32),
            "policy_head": tf.io.FixedLenFeature([1858], tf.float32),
        }
        example_proto = tf.io.parse_single_example(example_proto, feature_description)
        return example_proto["image"], {"value_head": example_proto["value_head"], "policy_head": example_proto["policy_head"]}
    
    def load_dataset(filenames):
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed
        dataset = tf.data.TFRecordDataset(filenames)  # automatically interleaves reads from multiple files
        dataset = dataset.with_options(ignore_order)  # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(partial(read_tfrecord), num_parallel_calls=tf.data.AUTOTUNE)
        return dataset
    
    def get_dataset(filenames, batch_size, buffer_size=4096):
        dataset = load_dataset(filenames)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        return dataset

    epoch_from = training_info['current_epoch']
    epoch_to = epoch_from + epochs

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(config['project_dir'], 'data', 'models', 'checkpoint.model.keras'),
        verbose=0,
        save_freq=100,
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(config['project_dir'], 'data', 'logs', 'fit'),
        histogram_freq=5,
    )
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        lambda _: config['learning_rate'],
    )
    callbacks = [checkpoint_callback, tensorboard_callback, lr_callback]
    if os.path.exists(os.path.join(config['project_dir'], 'data', 'models', 'checkpoint.model.keras')):
        try:
            model = tf.keras.models.load_model(os.path.join(config['project_dir'], 'data', 'models', 'checkpoint.model.keras'))
        except:
            model = tf.keras.models.load_model(os.path.join(config['project_dir'], 'data', 'models', 'model.keras'))
    else:
        model = tf.keras.models.load_model(os.path.join(config['project_dir'], 'data', 'models', 'model.keras'))

    records_dir = os.path.join(config['project_dir'], 'data', 'train_data')
    records = os.listdir(records_dir)
    records.sort()
    latest_record = records[-1]
    tf_record = os.path.join(config['project_dir'], 'data', 'train_data', latest_record)

    model.fit(
        get_dataset(tf_record, batch_size, buffer_size=40960*2),
        initial_epoch=epoch_from,
        epochs=epoch_to,
        steps_per_epoch=1,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(os.path.join(config['project_dir'], 'data', 'models', 'model.keras'))


def train_model_process(config, training_info, batch_size, epochs):
    p = Process(target=train_model, args=(config, training_info, batch_size, epochs))
    p.start()
    p.join()

# TODO:
# - Implement loop/cycle style training
# - Implement safety checks eg. if data dir is not initialized properly etc.
if __name__ == "__main__":
    # Check data dir exists
    if not os.path.exists(os.path.join(trainingConfig['project_dir'], 'data')):
        print("Data directory not found. Please run 'chessbot init' first.")
        print("For more information, run 'chessbot init --help'")
        exit()
        
    parser = argparse.ArgumentParser(description='Run training for ChessBot.')
    parser.add_argument('--calc', action='store_true', help='Calculate the number of epochs, sts_tests, checkpoints etc. based on available samples.')
    parser.add_argument('--auto', action='store_true', help='Automatically train the model.')
    
    args = parser.parse_args()

    config = trainingConfig
    training_info = load_training_info(trainingConfig)

    curr_epoch = training_info['current_epoch']
    batch_size = config['batch_size']

    checkpoint_interval = config['checkpoint_interval']
    sts_test_interval = config['sts_test_interval']
    max_epochs_per_cycle = config['max_epochs_per_cycle']

    if args.calc:
        total_samples, sampled_samples = count_available_samples(config)
        epochs_possible = sampled_samples // batch_size
        next_checkpoint_at = ((curr_epoch // checkpoint_interval) + 1) * checkpoint_interval
        next_sts_test_at = ((curr_epoch // sts_test_interval) + 1) * sts_test_interval
        print(f"Calculations based on available samples with batch_size: {batch_size}")
        print(f"Total samples: {total_samples}")
        print(f"Total samples (after sampling): {sampled_samples}")
        print(f"Current epoch: {curr_epoch}")
        print(f"Epochs possible: {epochs_possible}")
        print(f"Next checkpoint at: {next_checkpoint_at}")
        print(f"Next STS test at: {next_sts_test_at}")
        exit()

    if args.auto:
        try:
            total_samples, sampled_samples = count_available_samples(config)
            epochs_possible = sampled_samples // batch_size
            
            assert epochs_possible > 0, "Not enough samples available for training."

            next_checkpoint_at = ((curr_epoch // checkpoint_interval) + 1) * checkpoint_interval
            next_sts_test_at = ((curr_epoch // sts_test_interval) + 1) * sts_test_interval

            epochs_until_checkpoint = next_checkpoint_at - curr_epoch
            epochs_until_sts_test = next_sts_test_at - curr_epoch

            epochs_until_stop = min(epochs_until_checkpoint, epochs_until_sts_test)
            epochs = min(epochs_possible, epochs_until_stop)
            if max_epochs_per_cycle > 0:
                epochs = min(epochs, max_epochs_per_cycle)

            num_samples = batch_size * epochs

            samples = load_samples(config, round(num_samples / config['sampling_ratio']), num_samples)
            if len(samples) != num_samples:
                raise ValueError(f"Not enough samples loaded. Expected {num_samples} but got {len(samples)}")
            
            create_tf_record(config, samples)
            samples = None  # Free up memory

            print(f"Training model for {epochs} epochs with {num_samples} samples...")
            train_model_process(config, training_info, batch_size, epochs)
            
            curr_epoch += epochs

            if curr_epoch >= next_checkpoint_at:
                # Save checkpoint model
                print("Saving checkpoint model...")
                update_trt_model_process(config, model_version="latest", precision_mode="FP16", build_model=True)
            if curr_epoch >= next_sts_test_at:
                # Run STS testing
                if next_checkpoint_at != next_sts_test_at:
                    #Create new temporary model for STS testing
                    print("Creating temporary model for STS testing...")
                    update_trt_model_process(config, model_version="tmp", precision_mode="FP16", build_model=True)
                # Run STS testing
                model_version = "tmp" if next_checkpoint_at != next_sts_test_at else "latest"
                print(f"Running STS testing with '{model_version}'...")
                do_sts_test(config, current_epoch=curr_epoch, model_version=model_version)

            training_info['current_epoch'] = curr_epoch
            training_info['total_samples'] = training_info.get('total_samples', 0) + num_samples
            save_training_info(config, training_info)
        except Exception as e:
            print(f"Error: {e}")
            
            
            
            