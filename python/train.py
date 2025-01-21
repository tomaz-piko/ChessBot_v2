from .config import TrainingConfig
from .play_game import play_game
import numpy as np
from time import time
from multiprocessing import Process, Manager
from datetime import datetime
import pickle
import json
import os
import sys
from functools import partial
import gc
from .sts_test import do_strength_test

def lr_scheduler(epoch, lr):
    config = TrainingConfig()
    return config.learning_rate

def play_n_games(pid, config, games_count):
    import tensorflow as tf
    if config.allow_gpu_growth:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    from trt_funcs import load_trt_checkpoint_latest

    # Load model
    trt_func, _ = load_trt_checkpoint_latest()
    # Play games until GAMES_PER_CYCLE reached for all processes combined
    current_game = 0
    while games_count.value < config.games_per_cycle:
        t1 = time()
        images, (terminal_values, visit_counts), summary = play_game(config, trt_func)
        t2 = time()
        print(f"Pid: {pid}. Game {current_game}. {summary}. In {t2-t1:.2f} seconds.")
        games_count.value += 1
        timestamp = datetime.now().strftime('%H:%M:%S')
        for i, (image, terminal_value, visit_count) in enumerate(zip(images, terminal_values, visit_counts)):
            # Save each position as npz
            np.savez_compressed(f"{config.self_play_positions_dir}/{pid}_{current_game}_{i}-{timestamp}.npz", 
                                image=image,
                                terminal_value=[terminal_value],
                                visit_count=visit_count)
        del images, terminal_values, visit_counts
        current_game += 1
        gc.collect()

def self_play(config):
    processes = {}
    with Manager() as manager:
        games_count = manager.Value('i', 0)
        for pid in range(config.num_actors):
            p = Process(target=play_n_games, args=(pid, config, games_count,))
            p.start()
            processes[pid] = p

        for p in processes.values():
            p.join()

def prepare_data(config):
    import tensorflow as tf
    from gameimage import convert_to_model_input
    
    # All generated positions
    positions = os.listdir(config.self_play_positions_dir)
    assert len(positions) > config.batch_size, "Not enough positions to generate training data. Run self_play first."

    num_samples = int(len(positions) * config.sampling_ratio)
    epochs_count = num_samples // config.batch_size
    num_samples = epochs_count * config.batch_size
    positions_chosen = np.random.default_rng().choice(len(positions), size=num_samples, replace=False)

    # Generate TFRecords
    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    with tf.io.TFRecordWriter(f"{config.training_records_dir}/{timestamp}-{num_samples}.tfrecords") as writer:
        for i in positions_chosen:
            data_np = np.load(f"{config.self_play_positions_dir}/{positions[i]}")
            image = convert_to_model_input(data_np["image"])
            image_features = tf.train.Feature(float_list=tf.train.FloatList(value=image.flatten()))
            terminal_value_features = tf.train.Feature(float_list=tf.train.FloatList(value=data_np["terminal_value"]))
            visit_count_features = tf.train.Feature(float_list=tf.train.FloatList(value=data_np["visit_count"]))

            example = tf.train.Example(features=tf.train.Features(feature={
                "image": image_features,
                "value_head": terminal_value_features,
                "policy_head": visit_count_features           
            }))
            writer.write(example.SerializeToString())

    # Clean up old records
    records = os.listdir(config.training_records_dir)
    records.sort()
    while len(records) > 1:
        os.remove(f"{config.training_records_dir}/{records.pop(0)}")

    for position in positions:
        try:
            if config.self_play_positions_backup_dir:
                os.rename(f"{config.self_play_positions_dir}/{position}", f"{config.self_play_positions_backup_dir}/{position}")
            else:
                os.remove(f"{config.self_play_positions_dir}/{position}")
        except:
            continue

    print("--- DATA PREPERATION STATISTICS ---")
    print("===================================")
    print(f"Picked {num_samples} samples out of {len(positions)} available positions.")
    print(f"Generated data for {epochs_count} epochs with {config.batch_size} samples each.")
    print("===================================")


def train(config, current_step):
    import tensorflow as tf

    def read_tfrecord(example_proto):
        feature_description = {
            "image": tf.io.FixedLenFeature([config.image_shape[0], 8, 8], tf.float32),
            "value_head": tf.io.FixedLenFeature([1], tf.float32),
            "policy_head": tf.io.FixedLenFeature([config.num_actions], tf.float32),
        }
        example_proto = tf.io.parse_single_example(example_proto, feature_description)
        return example_proto["image"], (example_proto["value_head"], example_proto["policy_head"])
    
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

    # Get latest record file
    records = os.listdir(config.training_records_dir)
    records.sort()
    latest_record = records[-1]
    print(f"Scheduling training on {latest_record}.")

    num_samples = int(latest_record.split("-")[-1].split(".")[0])
    num_epochs = int(num_samples / config.batch_size)
    print(f"Training on {num_samples} samples for {num_epochs} epochs.")
    

    if os.path.exists(config.training_info_stats):
        with open(config.training_info_stats, "rb") as f:
            training_info_stats = json.load(f)
    else:
        training_info_stats = {
            "current_epoch": 0,
            "last_finished_step": 0,
            "samples_processed": 0,
        }
    
    epoch_from = training_info_stats["current_epoch"]
    epoch_to = training_info_stats["current_epoch"] + num_epochs

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{config.keras_checkpoint_dir}/checkpoint.model.keras",
        verbose=0,
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{config.tensorboard_log_dir}/fit/{config.model_name}", histogram_freq=1)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    callbacks = [checkpoint_callback, tensorboard_callback, lr_callback]

    if os.path.exists(f"{config.keras_checkpoint_dir}/checkpoint.model.keras"):
        try:
            model = tf.keras.models.load_model(f"{config.keras_checkpoint_dir}/checkpoint.model.keras")
        except:
            model = tf.keras.models.load_model(f"{config.keras_checkpoint_dir}/model.keras")
    else:
        model = tf.keras.models.load_model(f"{config.keras_checkpoint_dir}/model.keras")

    model.fit(
        get_dataset(f"{config.training_records_dir}/{latest_record}", config.batch_size, buffer_size=num_samples),
        initial_epoch=epoch_from,
        epochs=epoch_to,
        steps_per_epoch=1,
        callbacks=callbacks,
        verbose=1)
    
    # Update tracked training info
    training_info_stats["current_epoch"] = epoch_to
    training_info_stats["last_finished_step"] = current_step
    training_info_stats["samples_processed"] += num_samples
    with open(config.training_info_stats, "w") as f:
        json.dump(training_info_stats, f)
    
    # Save model
    model.save(f"{config.keras_checkpoint_dir}/model.keras")

def save(config, tmp=False):
    import tensorflow as tf
    if config.allow_gpu_growth:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    from trt_funcs import save_trt_model

    if not tmp:
        trt_save_path = f"{config.trt_checkpoint_dir}/{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}/saved_model"
    else:
        trt_save_path = f"{config.tmp_trt_checkpoint_dir}/saved_model"
        
    keras_model_path = f"{config.keras_checkpoint_dir}/model.keras"
    save_trt_model(keras_model_path, trt_save_path, config.trt_precision_mode)

def perform_sts_test(step, config):
    import os
    import tensorflow as tf

    # If checkpoint and sts test interval are not the same we make a temporary trt model
    # At certain points of training it can happen that we could be making a checkpoint and a tmp model at the same time
    # e.g. check_int = 9, sts_int = 3, at step 9 we would make a checkpoint and a tmp model which is a waste of resources
    if config.checkpoint_interval != config.sts_test_interval or step % config.checkpoint_interval != 0:
        model_path = "tmp"
    else:
        model_path = "lts"

    sts_rating = do_strength_test(config.sts_time_limit, config.sts_num_agents, model_path=model_path, save_results=True)
    count = len(os.listdir(config.sts_results_dir))
    # Save to tensorboard
    sts_summary_writter = tf.summary.create_file_writer(f"{config.tensorboard_log_dir}/rating/{config.model_name}")
    with sts_summary_writter.as_default():
        tf.summary.scalar("ELO Rating", sts_rating, step=count)

if __name__ == "__main__":
    config = TrainingConfig()
    skip_selfplay_step = False
    args = sys.argv[1:]
    if len(args) > 0:
        config.num_actors = int(args[0])
    if len(args) > 1:
        if "--skip-selfplay-step" in args:
            skip_selfplay_step = True

    initial_step = 0
    if os.path.exists(config.training_info_stats):
        with open(config.training_info_stats, "rb") as f:
            training_info_stats = json.load(f)
        initial_step = training_info_stats["last_finished_step"] + 1

    i = initial_step
    while i <= config.num_cycles:
        # Generate N games
        print(f"Scheduling self play training step {i}.")
        if not skip_selfplay_step: # Todo make option to skip multiple self play steps
            self_play(config)
        else:
            skip_selfplay_step = False

        # Save as tensor records?
        print(f"Preparing data for training.")
        prepare_data(config)

        # Train on those games
        p = Process(target=train, args=(config, i))
        p.start()
        p.join()

        # Save trt model
        if i > 0 and i % config.checkpoint_interval == 0:
            print(f"Converting model to TRT format.")
            p = Process(target=save, args=(config,))
            p.start()
            p.join()

        if i % config.sts_test_interval == 0:
            print(f"Converting model to TRT format.")
            p = Process(target=save, args=(config, True))
            p.start()
            p.join()
            print(f"Running STS test.")
            p = Process(target=perform_sts_test, args=(i, config,))
            p.start()
            p.join()

        i += 1    