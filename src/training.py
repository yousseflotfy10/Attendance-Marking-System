from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess_input
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess_input
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception:
    keras = None
    layers = None
    MobileNetV2 = None
    EfficientNetB0 = None
    VGG16 = None
    ImageDataGenerator = None
    mobilenet_preprocess_input = None
    efficientnet_preprocess_input = None
    vgg16_preprocess_input = None

from config import (
    DEFAULT_RECOGNITION_BACKBONE,
    FACE_INPUT_SIZE,
    LABEL_MAP_PATH,
    RECOGNITION_BACKBONES,
    RECOGNITION_METADATA_PATH,
    RECOGNITION_MODEL_PATH,
)


MODEL_SPECS = {
    "mobilenetv2": {
        "builder": MobileNetV2,
        "preprocess": mobilenet_preprocess_input,
        "display_name": "MobileNetV2",
    },
    "efficientnetb0": {
        "builder": EfficientNetB0,
        "preprocess": efficientnet_preprocess_input,
        "display_name": "EfficientNetB0",
    },
    "vgg16": {
        "builder": VGG16,
        "preprocess": vgg16_preprocess_input,
        "display_name": "VGG16",
    },
}


def _require_tensorflow() -> None:
    if keras is None or layers is None or ImageDataGenerator is None:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise RuntimeError(
            "TensorFlow is not available in the current Python interpreter "
            f"({python_version}). Use a TensorFlow-supported Python version "
            "(recommended: 3.10-3.13) and install TensorFlow there."
        )


def _build_base_model(backbone_name: str):
    spec = MODEL_SPECS.get(backbone_name)
    if spec is None or spec["builder"] is None:
        raise RuntimeError(f"Unsupported backbone: {backbone_name}")
    return spec["builder"](include_top=False, weights="imagenet", input_shape=(FACE_INPUT_SIZE[0], FACE_INPUT_SIZE[1], 3))


def build_training_model(num_classes: int, backbone_name: str = "mobilenetv2", output_dtype: str = "float32") -> Any:
    _require_tensorflow()
    base_model = _build_base_model(backbone_name)
    base_model.trainable = False

    inputs = keras.Input(shape=(FACE_INPUT_SIZE[0], FACE_INPUT_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype=output_dtype)(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def create_data_generators(dataset_dir: Path, batch_size: int = 32, validation_split: float = 0.2):
    _require_tensorflow()
    train_generator = ImageDataGenerator(
        validation_split=validation_split,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        fill_mode="nearest",
    )

    validation_generator = ImageDataGenerator(validation_split=validation_split)

    train_flow = train_generator.flow_from_directory(
        str(dataset_dir),
        target_size=FACE_INPUT_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
    )
    validation_flow = validation_generator.flow_from_directory(
        str(dataset_dir),
        target_size=FACE_INPUT_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )
    return train_flow, validation_flow


def _select_preprocess_input(backbone_name: str):
    spec = MODEL_SPECS.get(backbone_name)
    if spec is None:
        raise RuntimeError(f"Unsupported backbone: {backbone_name}")
    preprocess_input = spec["preprocess"]
    if preprocess_input is None:
        raise RuntimeError(f"TensorFlow preprocessing is unavailable for backbone: {backbone_name}")
    return preprocess_input


def _attach_preprocessing(datagen: Any, backbone_name: str) -> Any:
    preprocess_input = _select_preprocess_input(backbone_name)
    datagen.preprocessing_function = preprocess_input
    return datagen


def _training_callbacks() -> list[Any]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.25,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def train_single_backbone(
    dataset_dir: Path,
    backbone_name: str,
    epochs: int = 15,
    batch_size: int = 32,
    mixed_precision: bool = False,
    use_distribute: bool = False,
    workers: int = 4,
):
    _require_tensorflow()
    import tensorflow as tf
    from tensorflow import keras as _keras
    import numpy as _np
    from sklearn.model_selection import train_test_split
    import os
    from PIL import Image

    dataset_dir = Path(dataset_dir)
    
    # Load all images in memory and split them (train:60%, val:20%, test:20%)
    print("Loading dataset...")
    all_images = []
    all_labels = []
    class_to_idx = {}
    idx_to_class = {}
    
    class_idx = 0
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        class_to_idx[class_name] = class_idx
        idx_to_class[class_idx] = class_name
        
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                continue
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((FACE_INPUT_SIZE[0], FACE_INPUT_SIZE[1]))
                img_array = _np.array(img)
                all_images.append(img_array)
                all_labels.append(class_idx)
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")
                continue
        class_idx += 1
    
    all_images = _np.array(all_images, dtype='uint8')
    all_labels = _np.array(all_labels, dtype='int32')
    
    num_classes = len(class_to_idx)
    print(f"Loaded {len(all_images)} images from {num_classes} classes")
    
    # Split: 60% train, 20% val, 20% test (avoid data leakage by splitting first)
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_images, all_labels, test_size=0.4, random_state=42, stratify=all_labels
    )
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
    
    # Optionally enable mixed precision
    if mixed_precision:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled: policy=mixed_float16")
        except Exception:
            print("Mixed precision requested but unavailable in this TensorFlow build.")

    # Optionally use a distribution strategy
    strategy = None
    if use_distribute:
        try:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using distribution strategy: {strategy}")
        except Exception:
            strategy = None
            print("Requested distribution strategy not available; continuing without it.")
    
    # Preprocess function for the backbone
    preprocess_input = _select_preprocess_input(backbone_name)
    
    # Create data generators with augmentation for training only
    train_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        fill_mode="nearest",
        preprocessing_function=preprocess_input,
    )
    
    val_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    # Convert labels to one-hot
    train_labels_categorical = _keras.utils.to_categorical(train_labels, num_classes)
    val_labels_categorical = _keras.utils.to_categorical(val_labels, num_classes)
    test_labels_categorical = _keras.utils.to_categorical(test_labels, num_classes)
    
    # Create generators from arrays
    train_flow = train_generator.flow(
        train_imgs, train_labels_categorical,
        batch_size=batch_size, shuffle=True, seed=42
    )
    val_flow = val_generator.flow(
        val_imgs, val_labels_categorical,
        batch_size=batch_size, shuffle=False
    )
    test_flow = test_generator.flow(
        test_imgs, test_labels_categorical,
        batch_size=batch_size, shuffle=False
    )
    
    # Build model
    output_dtype = "float32"
    if strategy is not None:
        with strategy.scope():
            model = build_training_model(num_classes=num_classes, backbone_name=backbone_name, output_dtype=output_dtype)
    else:
        model = build_training_model(num_classes=num_classes, backbone_name=backbone_name, output_dtype=output_dtype)

    # Compute class weights
    try:
        unique, counts = _np.unique(train_labels, return_counts=True)
        total = counts.sum()
        class_weight = {int(u): float(total) / (num_classes * count) for u, count in zip(unique, counts)}
    except Exception:
        class_weight = None

    callbacks = [
        _keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        _keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    # Stage 1: train classifier head
    head_epochs = max(3, min(epochs, max(1, epochs // 3)))
    print(f"Stage 1/2: training classifier head for {head_epochs} epochs")
    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=head_epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    # Stage 2: fine-tune backbone tail
    fine_tune_epochs = max(0, epochs - head_epochs)
    if fine_tune_epochs:
        try:
            base_model = model.layers[1]
            base_model.trainable = True
            trainable_tail_layers = 30 if backbone_name == DEFAULT_RECOGNITION_BACKBONE else 20
            fine_tune_at = max(0, len(base_model.layers) - trainable_tail_layers)
            for layer_index, layer in enumerate(base_model.layers):
                if layer_index < fine_tune_at or isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False

            model.compile(optimizer=_keras.optimizers.Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
            print(f"Stage 2/2: fine-tuning last {trainable_tail_layers} backbone layers for {fine_tune_epochs} epochs")
            history = model.fit(
                train_flow,
                validation_data=val_flow,
                epochs=fine_tune_epochs,
                callbacks=callbacks,
                class_weight=class_weight,
            )
        except Exception:
            print("Fine-tuning failed; proceeding with the trained head model.")

    # Evaluate on validation set
    val_evaluation = model.evaluate(val_flow, verbose=0)
    val_accuracy = float(val_evaluation[1]) if len(val_evaluation) > 1 else 0.0
    
    # Evaluate on held-out test set (NO DATA LEAKAGE - separate from train/val)
    print("\nEvaluating on test set...")
    test_evaluation = model.evaluate(test_flow, verbose=0)
    test_accuracy = float(test_evaluation[1]) if len(test_evaluation) > 1 else 0.0
    
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy (final - unbiased): {test_accuracy:.4f}")
    
    # Create a simple object to return train_flow attributes
    class FlowInfo:
        def __init__(self, num_classes, class_indices):
            self.num_classes = num_classes
            self.class_indices = class_indices
    
    train_flow_info = FlowInfo(num_classes, class_to_idx)
    
    return history, model, train_flow_info, test_accuracy, val_accuracy


def train_recognition_model(
    dataset_dir: Path,
    model_output_path: Path = RECOGNITION_MODEL_PATH,
    label_map_output_path: Path = LABEL_MAP_PATH,
    metadata_output_path: Path = RECOGNITION_METADATA_PATH,
    epochs: int = 15,
    batch_size: int = 32,
    backbone: str = DEFAULT_RECOGNITION_BACKBONE,
    mixed_precision: bool = False,
    use_distribute: bool = False,
    workers: int = 4,
):
    dataset_dir = Path(dataset_dir)
    model_output_path = Path(model_output_path)
    label_map_output_path = Path(label_map_output_path)
    metadata_output_path = Path(metadata_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    _require_tensorflow()
    if backbone != "auto" and backbone not in RECOGNITION_BACKBONES:
        raise RuntimeError(f"Backbone must be one of: auto, {', '.join(RECOGNITION_BACKBONES)}")

    candidates = RECOGNITION_BACKBONES if backbone == "auto" else (backbone,)
    best_result = None
    all_results = []

    for candidate_backbone in candidates:
        print(f"\n{'='*60}")
        print(f"Phase {candidates.index(candidate_backbone) + 1}/{len(candidates)}: Training {candidate_backbone.upper()}")
        print(f"{'='*60}")
        
        history, model, train_flow, test_accuracy, val_accuracy = train_single_backbone(
            dataset_dir,
            candidate_backbone,
            epochs=epochs,
            batch_size=batch_size,
            mixed_precision=mixed_precision,
            use_distribute=use_distribute,
            workers=workers,
        )
        
        # Save individual model
        model_file = model_output_path.parent / f"{candidate_backbone}_model.keras"
        model.save(model_file)
        print(f"Saved model: {model_file}")
        
        # Store results for comparison (use test accuracy for selection - no data leakage!)
        result_entry = {
            "backbone": candidate_backbone,
            "test_accuracy": float(test_accuracy),
            "validation_accuracy": float(val_accuracy),
            "num_classes": train_flow.num_classes,
            "model_file": str(model_file),
        }
        all_results.append(result_entry)
        
        # Track best result (select based on test accuracy - unbiased!)
        if best_result is None or test_accuracy > best_result["test_accuracy"]:
            best_result = {
                "history": history,
                "model": model,
                "train_flow": train_flow,
                "test_accuracy": test_accuracy,
                "validation_accuracy": val_accuracy,
                "backbone": candidate_backbone,
                "model_file": str(model_file),
            }

    assert best_result is not None
    
    # Save comparison results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    comparison_file = model_output_path.parent / "comparison_results.json"
    comparison_data = {
        "all_models": all_results,
        "best_model": {
            "backbone": best_result["backbone"],
            "test_accuracy": best_result["test_accuracy"],
            "validation_accuracy": best_result["validation_accuracy"],
            "model_file": best_result["model_file"],
        }
    }
    with comparison_file.open("w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    
    for entry in all_results:
        is_best = "BEST" if entry["backbone"] == best_result["backbone"] else ""
        print(f"{entry['backbone']:20s} | Test Accuracy: {entry['test_accuracy']:.4f} | Val Accuracy: {entry['validation_accuracy']:.4f} {is_best}")
    print(f"{'='*60}")
    print(f"Saved comparison: {comparison_file}")
    
    # Copy best model as the main recognition model
    best_result["model"].save(model_output_path)
    print(f"Set best model as: {model_output_path}")

    # Save label map and metadata
    label_map = {index: label for label, index in best_result["train_flow"].class_indices.items()}
    with label_map_output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(label_map, file_handle, indent=2)

    metadata = {
        "backbone": best_result["backbone"],
        "test_accuracy": best_result["test_accuracy"],
        "validation_accuracy": best_result["validation_accuracy"],
        "num_classes": best_result["train_flow"].num_classes,
        "class_indices": best_result["train_flow"].class_indices,
    }
    with metadata_output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2)

    return best_result["history"], best_result["model"], metadata
