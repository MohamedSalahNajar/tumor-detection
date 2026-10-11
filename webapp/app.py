"""Flask web application for tumor detection prototype."""

import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from threading import Thread

import torch
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image
from werkzeug.utils import secure_filename

from ai.breastmnist import BreastMNISTSplit
from ai.evaluate import collect_outputs, load_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global model cache
model_cache = {}
checkpoint_path = Path("runs/breastmnist_baseline_seed7/model.pt")


def get_model():
    """Load and cache the model."""
    if 'model' not in model_cache:
        device = torch.device("cpu")
        model, checkpoint = load_model(checkpoint_path, device)
        model_cache['model'] = model
        model_cache['checkpoint'] = checkpoint
    return model_cache['model'], model_cache['checkpoint']


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp'}


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/api/model/info')
def model_info():
    """Get model information."""
    try:
        model, checkpoint = get_model()
        return jsonify({
            'status': 'loaded',
            'checkpoint': str(checkpoint_path),
            'selected_epoch': checkpoint.get('selected_epoch'),
            'config': checkpoint.get('config', {})
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Run prediction on uploaded image."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Load and preprocess image
        image = Image.open(file.stream).convert('L')
        image = image.resize((64, 64), Image.LANCZOS)

        # Convert to tensor
        import numpy as np
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)

        # Get prediction
        model, _ = get_model()
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            probability = torch.sigmoid(output).item()

        prediction = int(probability >= 0.5)
        label = 'malignant' if prediction == 1 else 'normal/benign'

        return jsonify({
            'prediction': prediction,
            'label': label,
            'probability': probability,
            'confidence': probability if prediction == 1 else 1 - probability
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/results')
def get_results():
    """Get all experiment results."""
    results_dir = Path("runs")
    results = {}

    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        results[exp_name] = {'metrics': {}, 'config': {}}

        # Load test metrics if available
        metrics_file = exp_dir / "test_metrics.json"
        if metrics_file.exists():
            results[exp_name]['metrics'] = json.loads(metrics_file.read_text())

        # Load config if available
        config_file = exp_dir / "run_config.json"
        if config_file.exists():
            results[exp_name]['config'] = json.loads(config_file.read_text())

        # Load additional metrics from subdirectories
        for subdir in exp_dir.iterdir():
            if subdir.is_dir():
                sub_metrics = subdir / "test_metrics.json"
                if sub_metrics.exists():
                    results[exp_name][subdir.name] = json.loads(sub_metrics.read_text())

    return jsonify(results)


@app.route('/api/train', methods=['POST'])
def train():
    """Start training with provided parameters."""
    try:
        params = request.json

        # Extract training parameters
        epochs = params.get('epochs', 5)
        batch_size = params.get('batch_size', 16)
        learning_rate = params.get('learning_rate', 0.001)
        base_channels = params.get('base_channels', 8)
        blocks_per_stage = params.get('blocks_per_stage', 1)
        dropout_rate = params.get('dropout_rate', 0.3)
        weight_decay = params.get('weight_decay', 0.01)
        positive_class_weight = params.get('positive_class_weight', 1.0)
        seed = params.get('seed', 7)
        skip_test = params.get('skip_test', True)
        output = params.get('output', f'runs/custom_run_seed{seed}')

        # Build command
        cmd = [
            sys.executable, '-m', 'ai.train',
            '--dataset', 'breastmnist',
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--learning-rate', str(learning_rate),
            '--base-channels', str(base_channels),
            '--blocks-per-stage', str(blocks_per_stage),
            '--dropout-rate', str(dropout_rate),
            '--weight-decay', str(weight_decay),
            '--positive-class-weight', str(positive_class_weight),
            '--seed', str(seed),
            '--output', output
        ]

        if skip_test:
            cmd.append('--skip-test')

        # Run training in background thread
        def run_training():
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Training failed: {e.stderr}")

        thread = Thread(target=run_training)
        thread.daemon = True
        thread.start()

        return jsonify({
            'status': 'started',
            'output': output,
            'parameters': params
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tests')
def run_tests():
    """Run the test suite."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
            capture_output=True,
            text=True,
            timeout=60
        )

        return jsonify({
            'status': 'completed',
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        })
    except subprocess.TimeoutExpired:
        return jsonify({'status': 'error', 'message': 'Tests timed out'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
