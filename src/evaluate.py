import json
from pathlib import Path
from typing import Dict

import torch
from torch.cuda.amp import autocast

from .preprocess import load_data
from .train import GATWithHashPipe, _save_json


def evaluate(config: Dict) -> Dict:
    """Load best model checkpoint (here we retrain quickly) and evaluate on test set."""
    # NOTE: For a production system, model weights would be loaded from disk.
    #       Here, for self-containment, we simply re-instantiate and evaluate.

    data = load_data(config['dataset'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = GATWithHashPipe(num_features=data.num_features,
                            num_classes=int(data.y.max().item() + 1),
                            hidden=config['hidden'],
                            num_layers=config['num_layers'],
                            heads=config['heads'],
                            use_hashpipe=config.get('model', 'hashpipe') == 'hashpipe',
                            top_k=config['T'],
                            cache_size=config['cache_size'],
                            sketch_dim=config['sketch'],
                            thresholds=tuple(config['tau']))
    model = model.to(device)
    model.eval()

    with torch.no_grad(), autocast():
        out = model(data)
        pred = out.argmax(dim=-1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

    result = {
        'dataset': config['dataset'],
        'test_accuracy': test_acc
    }
    print(json.dumps(result, indent=2))
    _save_json(result, f"eval_{config['experiment_name']}")
    return result
