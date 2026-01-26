"""
Experiment Tracking Integration

Provides integration with experiment tracking platforms (Weights & Biases).
Allows logging of hyperparameters, metrics, and artifacts for reproducible research.

Usage:
    # In entry.py or train_loop.py
    from experiment_tracking import ExperimentTracker
    
    tracker = ExperimentTracker(config, enable_wandb=True)
    tracker.log_metrics({'train_loss': 0.5, 'step': 100})
    tracker.finish()
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import json


class ExperimentTracker:
    """
    Unified interface for experiment tracking.
    
    Supports:
    - Weights & Biases (wandb)
    - Local JSON logging (fallback)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        project_name: str = "kerneldev",
        experiment_name: Optional[str] = None,
        enable_wandb: bool = True,
        wandb_entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            config: Full configuration dictionary
            project_name: Name of the project (for wandb)
            experiment_name: Name of this experiment run (auto-generated if None)
            enable_wandb: Whether to use Weights & Biases
            wandb_entity: W&B team/user name (optional)
            tags: List of tags for the experiment
            notes: Optional notes about the experiment
        """
        self.config = config
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.enable_wandb = enable_wandb
        self.wandb_entity = wandb_entity
        self.tags = tags or []
        self.notes = notes
        
        self.wandb = None
        self.wandb_run = None
        self.local_log = []
        
        self._initialize()
    
    def _initialize(self):
        """Initialize tracking backend(s)."""
        # Try to import and initialize wandb
        if self.enable_wandb:
            try:
                import wandb as wandb_module
                self.wandb = wandb_module
                
                # Check if wandb is configured
                if not self.wandb.api.api_key:
                    print("⚠️  W&B API key not found. Run 'wandb login' or set WANDB_API_KEY.")
                    print("   Falling back to local logging only.")
                    self.enable_wandb = False
                else:
                    self._init_wandb()
            except ImportError:
                print("⚠️  Weights & Biases not installed. Install with: pip install wandb")
                print("   Falling back to local logging only.")
                self.enable_wandb = False
        
        # Local logging is always enabled as fallback
        print(f"📊 Experiment tracking initialized")
        print(f"   - W&B: {'✓ enabled' if self.enable_wandb else '✗ disabled'}")
        print(f"   - Local JSON: ✓ enabled")
    
    def _init_wandb(self):
        """Initialize Weights & Biases."""
        # Prepare wandb config by flattening nested config
        wandb_config = self._flatten_config(self.config)
        
        # Initialize wandb run
        self.wandb_run = self.wandb.init(
            project=self.project_name,
            entity=self.wandb_entity,
            name=self.experiment_name,
            config=wandb_config,
            tags=self.tags,
            notes=self.notes,
            resume='allow'  # Allow resuming if run with same name exists
        )
        
        print(f"   - W&B Run: {self.wandb_run.name}")
        print(f"   - W&B URL: {self.wandb_run.url}")
    
    def _flatten_config(self, config: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested config dictionary for wandb."""
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """
        Log metrics to tracking backend(s).
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number
            commit: Whether to commit to wandb (allows batching if False)
        """
        # Add step to metrics if provided
        if step is not None and 'step' not in metrics:
            metrics['step'] = step
        
        # Log to wandb
        if self.enable_wandb and self.wandb_run:
            try:
                self.wandb.log(metrics, step=step, commit=commit)
            except Exception as e:
                print(f"⚠️  Failed to log to W&B: {e}")
        
        # Log locally
        self.local_log.append({
            'metrics': metrics.copy(),
            'step': step
        })
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """
        Log hyperparameters (called after initialization to update config).
        
        Args:
            params: Dictionary of hyperparameter name -> value
        """
        if self.enable_wandb and self.wandb_run:
            try:
                self.wandb_run.config.update(params)
            except Exception as e:
                print(f"⚠️  Failed to update W&B config: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "model", name: Optional[str] = None):
        """
        Log an artifact (file) to tracking backend.
        
        Args:
            artifact_path: Path to the artifact file
            artifact_type: Type of artifact (model, dataset, checkpoint, etc.)
            name: Name for the artifact (defaults to filename)
        """
        if self.enable_wandb and self.wandb_run:
            try:
                artifact = self.wandb.Artifact(
                    name=name or Path(artifact_path).stem,
                    type=artifact_type
                )
                artifact.add_file(artifact_path)
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                print(f"⚠️  Failed to log artifact to W&B: {e}")
    
    def save_local_log(self, output_path: str = "experiment_log.json"):
        """
        Save local log to JSON file.
        
        Args:
            output_path: Path to save the log file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'config': self.config,
                'project_name': self.project_name,
                'experiment_name': self.experiment_name,
                'metrics': self.local_log
            }, f, indent=2)
        
        return str(output_path)
    
    def watch_model(self, model, log_freq: int = 100, log_graph: bool = False):
        """
        Watch model for gradient and parameter tracking.
        
        Args:
            model: PyTorch model to watch
            log_freq: Frequency of logging (every N steps)
            log_graph: Whether to log computation graph
        """
        if self.enable_wandb and self.wandb_run:
            try:
                self.wandb.watch(model, log_freq=log_freq, log_graph=log_graph)
            except Exception as e:
                print(f"⚠️  Failed to watch model in W&B: {e}")
    
    def finish(self):
        """Finish the experiment and cleanup."""
        # Save local log
        if self.local_log:
            checkpoint_dir = self.config.get('training', {}).get('checkpoint_dir', 'checkpoints')
            log_path = Path(checkpoint_dir) / 'experiment_log.json'
            saved_path = self.save_local_log(log_path)
            print(f"📝 Experiment log saved to: {saved_path}")
        
        # Finish wandb run
        if self.enable_wandb and self.wandb_run:
            try:
                self.wandb_run.finish()
                print("✅ W&B run finished")
            except Exception as e:
                print(f"⚠️  Failed to finish W&B run: {e}")
    
    @property
    def run_id(self) -> Optional[str]:
        """Get the run ID (wandb run id or generated id)."""
        if self.enable_wandb and self.wandb_run:
            return self.wandb_run.id
        return self.experiment_name
    
    @property
    def run_url(self) -> Optional[str]:
        """Get the run URL (wandb only)."""
        if self.enable_wandb and self.wandb_run:
            return self.wandb_run.url
        return None


def create_experiment_tracker(
    config: Dict[str, Any],
    enable: bool = True,
    project_name: str = "kerneldev",
    experiment_name: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Optional[ExperimentTracker]:
    """
    Factory function to create experiment tracker.
    
    Args:
        config: Full configuration dictionary
        enable: Whether to enable experiment tracking
        project_name: Name of the project
        experiment_name: Name of this experiment
        tags: List of tags
    
    Returns:
        ExperimentTracker instance or None if disabled
    """
    if not enable:
        return None
    
    # Check config for wandb settings
    wandb_config = config.get('experiment_tracking', {})
    enable_wandb = wandb_config.get('enable_wandb', True)
    wandb_entity = wandb_config.get('wandb_entity', None)
    
    # Auto-generate experiment name from config if not provided
    if experiment_name is None:
        model_cfg = config.get('model', {})
        training_cfg = config.get('training', {})
        experiment_name = f"dim{model_cfg.get('dim', 'unk')}_layers{model_cfg.get('n_layers', 'unk')}_lr{training_cfg.get('learning_rate', 'unk')}"
    
    # Auto-generate tags from config
    if tags is None:
        tags = []
        if 'precision' in config.get('training', {}):
            tags.append(f"precision:{config['training']['precision']}")
        if 'tasks' in config:
            tags.extend([f"task:{task}" for task in config['tasks'].keys()])
    
    return ExperimentTracker(
        config=config,
        project_name=project_name,
        experiment_name=experiment_name,
        enable_wandb=enable_wandb,
        wandb_entity=wandb_entity,
        tags=tags
    )
