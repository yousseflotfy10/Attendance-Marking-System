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

from .config import FACE_INPUT_SIZE, LABEL_MAP_PATH, RECOGNITION_BACKBONES, RECOGNITION_METADATA_PATH, RECOGNITION_MODEL_PATH


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


def build_training_model(num_classes: int, backbone_name: str = "mobilenetv2") -> Any:
    _require_tensorflow()
    base_model = _build_base_model(backbone_name)
    base_model.trainable = False

    inputs = keras.Input(shape=(FACE_INPUT_SIZE[0], FACE_INPUT_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
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


def train_single_backbone(
    dataset_dir: Path,
    backbone_name: str,
    epochs: int = 15,
    batch_size: int = 32,
):
    _require_tensorflow()
    train_generator = ImageDataGenerator(
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        fill_mode="nearest",
    )
    validation_generator = ImageDataGenerator(validation_split=0.2)
    _attach_preprocessing(train_generator, backbone_name)
    _attach_preprocessing(validation_generator, backbone_name)

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

    model = build_training_model(num_classes=train_flow.num_classes, backbone_name=backbone_name)
    history = model.fit(train_flow, validation_data=validation_flow, epochs=epochs)
    evaluation = model.evaluate(validation_flow, verbose=0)
    validation_accuracy = float(evaluation[1]) if len(evaluation) > 1 else 0.0
    return history, model, train_flow, validation_accuracy


def train_recognition_model(
    dataset_dir: Path,
    model_output_path: Path = RECOGNITION_MODEL_PATH,
    label_map_output_path: Path = LABEL_MAP_PATH,
    metadata_output_path: Path = RECOGNITION_METADATA_PATH,
    epochs: int = 15,
    batch_size: int = 32,
    backbone: str = "auto",
):
    _require_tensorflow()
    dataset_dir = Path(dataset_dir)
    model_output_path = Path(model_output_path)
    label_map_output_path = Path(label_map_output_path)
    metadata_output_path = Path(metadata_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    if backbone != "auto" and backbone not in RECOGNITION_BACKBONES:
        raise RuntimeError(f"Backbone must be one of: auto, {', '.join(RECOGNITION_BACKBONES)}")

    candidates = RECOGNITION_BACKBONES if backbone == "auto" else (backbone,)
    best_result = None
    all_results = []

    for candidate_backbone in candidates:
        print(f"\n{'='*60}")
        print(f"Phase {candidates.index(candidate_backbone) + 1}/{len(candidates)}: Training {candidate_backbone.upper()}")
        print(f"{'='*60}")
        
        history, model, train_flow, validation_accuracy = train_single_backbone(
            dataset_dir,
            candidate_backbone,
            epochs=epochs,
            batch_size=batch_size,
        )
        
        # Save individual model
        model_file = model_output_path.parent / f"{candidate_backbone}_model.keras"
        model.save(model_file)
        print(f"✓ Saved model: {model_file}")
        
        # Store results for comparison
        result_entry = {
            "backbone": candidate_backbone,
            "validation_accuracy": float(validation_accuracy),
            "num_classes": train_flow.num_classes,
            "model_file": str(model_file),
        }
        all_results.append(result_entry)
        
        # Track best result
        if best_result is None or validation_accuracy > best_result["validation_accuracy"]:
            best_result = {
                "history": history,
                "model": model,
                "train_flow": train_flow,
                "validation_accuracy": validation_accuracy,
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
            "validation_accuracy": best_result["validation_accuracy"],
            "model_file": best_result["model_file"],
        }
    }
    with comparison_file.open("w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    
    for entry in all_results:
        is_best = "✓ BEST" if entry["backbone"] == best_result["backbone"] else ""
        print(f"{entry['backbone']:20s} | Accuracy: {entry['validation_accuracy']:.4f} {is_best}")
    print(f"{'='*60}")
    print(f"✓ Saved comparison: {comparison_file}")
    
    # Copy best model as the main recognition model
    best_result["model"].save(model_output_path)
    print(f"✓ Set best model as: {model_output_path}")

    # Save label map and metadata
    label_map = {index: label for label, index in best_result["train_flow"].class_indices.items()}
    with label_map_output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(label_map, file_handle, indent=2)

    metadata = {
        "backbone": best_result["backbone"],
        "validation_accuracy": best_result["validation_accuracy"],
        "num_classes": best_result["train_flow"].num_classes,
        "class_indices": best_result["train_flow"].class_indices,
    }
    with metadata_output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2)

    return best_result["history"], best_result["model"], metadata

