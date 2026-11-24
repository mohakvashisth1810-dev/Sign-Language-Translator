"""
Machine learning model for sign language classification.
Uses geometric hand features + domain-shift augmentation + soft-voting
ensemble (two regularized MLPs + RBF SVC) trained on hand landmark data.
"""

import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
from landmarks import extract_features, extract_features_batch

DATASET_FILENAME = "asl_alphabet_landmarks.csv"
LABEL_COLUMN = "label"

MAX_TRAIN_LANDMARK_NORM = 8.5
AUGMENT_COPIES = 5
MINORITY_FLOOR = 180


def _build_classifier():
    """Construct the soft-voting MLP + RBF-SVC ensemble."""
    return VotingClassifier(
        estimators=[
            ("mlp1", MLPClassifier(
                hidden_layer_sizes=(256, 128),
                alpha=5e-3,
                max_iter=350,
                early_stopping=True,
                random_state=config.RANDOM_STATE,
            )),
            ("mlp2", MLPClassifier(
                hidden_layer_sizes=(192, 96),
                alpha=8e-3,
                max_iter=350,
                early_stopping=True,
                random_state=7,
            )),
            ("svc", SVC(
                C=8.0,
                gamma="scale",
                kernel="rbf",
                probability=True,
                random_state=config.RANDOM_STATE,
            )),
        ],
        voting="soft",
        weights=[2, 2, 2],
    )


def _augment_landmarks(X, y, rng, copies=AUGMENT_COPIES):
    """
    Generate realistic pose/depth/noise perturbations of training landmarks
    and balance minority classes so rare letters (e.g. M, N) are not
    underweighted relative to common ones.
    """
    Xs, ys = [X], [y]
    n_samples = len(X)
    for _ in range(copies):
        Xp = X.reshape(-1, config.NUM_LANDMARKS, config.LANDMARK_DIMENSIONS).copy()
        for (ax_a, ax_b), limit_deg in (((0, 1), 18), ((0, 2), 20), ((1, 2), 18)):
            rad = np.deg2rad(rng.uniform(-limit_deg, limit_deg, size=n_samples))
            cs, sn = np.cos(rad)[:, None], np.sin(rad)[:, None]
            va, vb = Xp[:, :, ax_a].copy(), Xp[:, :, ax_b].copy()
            Xp[:, :, ax_a] = va * cs - vb * sn
            Xp[:, :, ax_b] = va * sn + vb * cs
        Xp[:, :, 2] *= rng.uniform(0.5, 1.4, size=(n_samples, 1))
        Xp[:, :, 0] *= rng.uniform(0.92, 1.08, size=(n_samples, 1))
        Xp += rng.normal(0, 0.018, size=Xp.shape)
        # Re-normalize by wrist -> middle_mcp (index 9) to match normalize_landmarks()
        scale = np.linalg.norm(Xp[:, 9, :], axis=1, keepdims=True)
        Xp = Xp / np.maximum(scale[:, None, :], 1e-8)
        Xs.append(Xp.reshape(-1, config.TOTAL_FEATURES))
        ys.append(y)

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) < MINORITY_FLOOR:
            need = MINORITY_FLOOR - len(idx)
            pick = rng.choice(idx, size=need, replace=True)
            Xp = X[pick].reshape(-1, config.NUM_LANDMARKS, config.LANDMARK_DIMENSIONS).copy()
            Xp += rng.normal(0, 0.012, size=Xp.shape)
            scale = np.linalg.norm(Xp[:, 9, :], axis=1, keepdims=True)
            Xp = Xp / np.maximum(scale[:, None, :], 1e-8)
            Xs.append(Xp.reshape(-1, config.TOTAL_FEATURES))
            ys.append(np.full(need, cls, dtype=y.dtype))

    return np.vstack(Xs), np.concatenate(ys)


class SignLanguageModel:
    """Geometric-feature ensemble classifier for ASL alphabet signs (26 classes)"""

    def __init__(self):
        """Initialize model"""
        self.model = _build_classifier()
        self.scaler = StandardScaler()
        self._label_encoder = LabelEncoder()
        self.label_encoder = {}
        self.label_decoder = {}
        self.offline_test_accuracy = None
        self.offline_macro_f1 = None
        self.test_samples = None
        self.train_samples = None
        self.num_features = None
        self.model_type = "VotingClassifier(MLP(256,128), MLP(192,96), SVC(rbf))"
        self.is_trained = False

    def build_model(self, num_classes):
        """
        Log model architecture.

        Args:
            num_classes: Number of sign classes
        """
        print("\nModel Architecture:")
        print(f"{self.model_type}")
        print(f"- Classes: {num_classes}")
        print(f"- Engineered Features: {self.num_features or 'dynamic'}")

    def load_data(self, data_dir=config.PROCESSED_DATA_DIR):
        """
        Load landmark data from CSV files

        Args:
            data_dir: Directory containing CSV files

        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        import pandas as pd      # training only; keeps it out of the web runtime

        csv_path = os.path.join(data_dir, DATASET_FILENAME)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        print(f"\nLoading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        feature_cols = [f"feature_{i}" for i in range(config.TOTAL_FEATURES)]
        missing = [c for c in [LABEL_COLUMN] + feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing expected columns: {missing[:5]}")

        X = df[feature_cols].values
        y = df[LABEL_COLUMN].values

        print(f"\nTotal samples: {len(X)}")
        print(f"Raw landmark dimensions: {X.shape}")
        print(f"Classes: {len(np.unique(y))}")

        return X, y

    def encode_labels(self, y):
        """
        Encode string labels to integers and create encoder/decoder

        Args:
            y: Array of string labels

        Returns:
            Encoded labels
        """
        y_encoded = self._label_encoder.fit_transform(y)

        # Mirror into plain dicts so metadata.json stays loadable by inference
        self.label_encoder = {
            str(label): int(idx)
            for idx, label in enumerate(self._label_encoder.classes_)
        }
        self.label_decoder = {
            int(idx): str(label)
            for idx, label in enumerate(self._label_encoder.classes_)
        }

        return y_encoded

    def prepare_data(self, X, y):
        """
        Split, clean training outliers, augment, extract geometric features, and scale.

        Args:
            X: Raw normalized landmark array (N, 63)
            y: Label array

        Returns:
            Tuple of (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        y_encoded = self.encode_labels(y)

        # Split before cleaning/augmentation so held-out evaluation is untouched
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=1 - config.TRAIN_TEST_SPLIT,
            random_state=config.RANDOM_STATE,
            stratify=y_encoded
        )

        # Drop extreme MediaPipe skeleton outliers from training split only
        train_norms = np.linalg.norm(
            X_train.reshape(-1, config.NUM_LANDMARKS, config.LANDMARK_DIMENSIONS),
            axis=(1, 2),
        )
        keep = train_norms <= MAX_TRAIN_LANDMARK_NORM
        X_train_clean = X_train[keep]
        y_train_clean = y_train[keep]

        rng = np.random.default_rng(config.RANDOM_STATE)
        X_train_aug, y_train_aug = _augment_landmarks(X_train_clean, y_train_clean, rng)

        F_train = extract_features_batch(X_train_aug)
        F_test = extract_features_batch(X_test)
        self.num_features = int(F_train.shape[1])
        self.train_samples = int(len(X_train))
        self.test_samples = int(len(X_test))

        X_train_scaled = self.scaler.fit_transform(F_train)
        X_test_scaled = self.scaler.transform(F_test)

        num_classes = len(self.label_encoder)

        print(f"\nData split:")
        print(f"Training samples: {len(X_train)} (cleaned={len(X_train_clean)}, augmented={len(X_train_aug)})")
        print(f"Test samples: {len(X_test)}")
        print(f"Engineered features: {self.num_features}")
        print(f"Number of classes: {num_classes}")

        return X_train_scaled, X_test_scaled, y_train_aug, y_test

    def train(self, data_dir=config.PROCESSED_DATA_DIR):
        """
        Train the model on collected data

        Args:
            data_dir: Directory containing CSV training data
        """
        print("\n" + "="*60)
        print("TRAINING SIGN LANGUAGE MODEL")
        print("="*60)

        try:
            X, y = self.load_data(data_dir)
            X_train, X_test, y_train, y_test = self.prepare_data(X, y)

            num_classes = len(self.label_encoder)
            self.build_model(num_classes)

            print("\nTraining model...")
            self.model.fit(X_train, y_train)
            print("✓ Training complete")

            print("\n" + "="*60)
            print("MODEL EVALUATION")
            print("="*60)

            y_pred = self.model.predict(X_test)

            accuracy = float(accuracy_score(y_test, y_pred))
            macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
            self.offline_test_accuracy = round(accuracy, 4)
            self.offline_macro_f1 = round(macro_f1, 4)
            print(f"\nTest Accuracy: {accuracy:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")

            class_indices = np.arange(num_classes)
            target_names = [self.label_decoder[i] for i in class_indices]

            print("\nClassification Report:")
            print(classification_report(
                y_test, y_pred,
                labels=class_indices,
                target_names=target_names,
                zero_division=0
            ))

            cm = confusion_matrix(y_test, y_pred, labels=class_indices)
            self._report_confusion_matrix(cm, target_names)

            os.makedirs(config.MODELS_DIR, exist_ok=True)
            model_path = config.MODEL_PATH.replace('.h5', '.pkl')
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, config.SCALER_PATH)

            print(f"\n✓ Model saved to {model_path}")
            print(f"✓ Scaler saved to {config.SCALER_PATH}")

            self.is_trained = True
            self.save_metadata()

            return accuracy

        except Exception as e:
            print(f"✗ Training error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _report_confusion_matrix(self, cm, target_names):
        """
        Display the 26-class confusion matrix and save it as CSV

        Args:
            cm: Confusion matrix array (num_classes x num_classes)
            target_names: Class names ordered by class index
        """
        print("\nConfusion Matrix (rows = true, cols = predicted):")
        header = "    " + " ".join(f"{n:>3}" for n in target_names)
        print(header)
        for name, row in zip(target_names, cm):
            print(f"{name:>3} " + " ".join(f"{v:>3}" for v in row))

        import pandas as pd      # training only; keeps it out of the web runtime

        os.makedirs(config.MODELS_DIR, exist_ok=True)
        cm_path = os.path.join(config.MODELS_DIR, "confusion_matrix.csv")
        pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(cm_path)
        print(f"\n✓ Confusion matrix saved to {cm_path}")

    def load_model(self, model_path=config.MODEL_PATH, scaler_path=config.SCALER_PATH):
        """
        Load pre-trained model and scaler

        Args:
            model_path: Path to saved model (.pkl format)
            scaler_path: Path to saved scaler
        """
        try:
            if model_path.endswith('.h5'):
                model_path = model_path.replace('.h5', '.pkl')

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            print(f"✓ Model loaded from {model_path}")
            print(f"✓ Scaler loaded from {scaler_path}")
        except FileNotFoundError as e:
            print(f"✗ Model file not found: {e}")
            raise

    def predict(self, landmarks):
        """
        Predict sign from normalized 63-value hand landmarks.

        Args:
            landmarks: Hand landmarks (numpy array or list)

        Returns:
            Tuple of (predicted_label, confidence)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load_model() first")

        try:
            if not isinstance(landmarks, np.ndarray):
                landmarks = np.array(landmarks)

            if len(landmarks.shape) > 1 and landmarks.shape[0] > 1:
                landmarks = landmarks[0]

            landmarks = landmarks.flatten()
            features = extract_features(landmarks)
            features_scaled = self.scaler.transform([features])

            probabilities = self.model.predict_proba(features_scaled)[0]
            best_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[best_idx])
            predicted_label = self.label_decoder[int(self.model.classes_[best_idx])]

            return predicted_label, confidence

        except Exception as e:
            print(f"✗ Prediction error: {e}")
            return None, 0.0

    def save_metadata(self):
        """Save model metadata (labels and evaluation metrics)"""
        metadata = {
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'model_type': self.model_type,
            'num_features': self.num_features,
            'offline_test_accuracy': self.offline_test_accuracy,
            'offline_macro_f1': self.offline_macro_f1,
            'train_samples': self.train_samples,
            'test_samples': self.test_samples,
        }
        import json
        metadata_path = os.path.join(config.MODELS_DIR, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print(f"✓ Metadata saved to {metadata_path}")

    def load_metadata(self):
        """Load model metadata (labels and evaluation metrics)"""
        import json
        metadata_path = os.path.join(config.MODELS_DIR, 'metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.label_encoder = metadata['label_encoder']
            self.label_decoder = {int(k): v for k, v in metadata['label_decoder'].items()}
            self.offline_test_accuracy = metadata.get('offline_test_accuracy')
            self.offline_macro_f1 = metadata.get('offline_macro_f1')
            self.num_features = metadata.get('num_features')
            self.model_type = metadata.get('model_type', self.model_type)
            self.train_samples = metadata.get('train_samples')
            self.test_samples = metadata.get('test_samples')
            print(f"✓ Metadata loaded from {metadata_path}")
        except FileNotFoundError:
            print(f"⚠ Metadata file not found: {metadata_path}")


def main():
    """Main training interface"""
    try:
        model = SignLanguageModel()

        print("\n" + "="*60)
        print("SIGN LANGUAGE MODEL TRAINER")
        print("="*60)
        print("\nOptions:")
        print("1. Train new model")
        print("2. Load existing model")
        print("3. Exit")
        print("="*60)

        choice = input("\nSelect option (1-3): ").strip()

        if choice == '1':
            model.train()
            model.save_metadata()
        elif choice == '2':
            model.load_model()
            model.load_metadata()
        elif choice == '3':
            print("Exiting...")
        else:
            print("Invalid option")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
