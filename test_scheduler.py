import unittest
import torch
import math
from train_loop import Trainer, TrainingConfig # Assuming TrainingConfig is accessible
from model import GPTModel # Mock or simple model

class TestSchedulerLogic(unittest.TestCase):

    def setUp(self):
        # Base config for the trainer
        self.config = TrainingConfig(
            learning_rate=1e-3,
            warmup_steps=10, # Small warmup for T_0 placeholder in Trainer.__init__
            eval_every=5, # Evaluate frequently for testing plateau
            plateau_monitor_metric='val_loss',
            plateau_patience=2, # Trigger plateau after 2 non-improving evaluations
            plateau_threshold=1e-4,
            plateau_mode='min',
            scheduler_T0_epoch_fraction=0.1, # T_0 will be 10% of steps_per_epoch
            scheduler_T_mult=1,
            num_epochs=5 # Not directly used by scheduler test but good for context
        )

        # Mock model and optimizer
        # Using a real simple model for testing optimizer state
        self.mock_model = GPTModel(vocab_size=10, dim=10, n_layers=1, n_heads=1, max_seq_len=10)
        self.optimizer = torch.optim.AdamW(self.mock_model.parameters(), lr=self.config.learning_rate)

        # Create a simplified Trainer instance for testing scheduler logic
        # We are not doing a full train, just testing the LR scheduling part
        self.trainer = Trainer(model=self.mock_model, config=self.config, data_builder=None)
        self.trainer.optimizer = self.optimizer # Ensure trainer uses this optimizer

        # Simulate steps_per_epoch (e.g., 100 steps per epoch)
        self.steps_per_epoch = 100
        self.trainer.steps_per_epoch = self.steps_per_epoch

        # Manually initialize the scheduler as it's done in trainer.train()
        new_T0 = math.ceil(self.steps_per_epoch * self.config.scheduler_T0_epoch_fraction)
        self.trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=new_T0,
            T_mult=self.config.scheduler_T_mult,
            eta_min=self.config.learning_rate * 0.1
        )
        self.trainer.initial_lr = self.config.learning_rate

        # Initialize plateau detection state for trainer
        self.trainer.plateau_best_metric_val = float('inf') if self.config.plateau_mode == 'min' else float('-inf')
        self.trainer.plateau_patience_counter = 0


    def simulate_evaluation(self, current_val_metric):
        # Simplified version of what happens in trainer.train_epoch() evaluation part
        # This function will call the plateau detection and potential scheduler reset

        improved = False
        if self.trainer.config.plateau_mode == 'min':
            if current_val_metric < self.trainer.plateau_best_metric_val - self.trainer.config.plateau_threshold:
                improved = True
        elif self.trainer.config.plateau_mode == 'max':
            if current_val_metric > self.trainer.plateau_best_metric_val + self.trainer.config.plateau_threshold:
                improved = True

        if improved:
            self.trainer.plateau_best_metric_val = current_val_metric
            self.trainer.plateau_patience_counter = 0
        else:
            self.trainer.plateau_patience_counter += 1

        if self.trainer.plateau_patience_counter >= self.trainer.config.plateau_patience:
            self.trainer.plateau_patience_counter = 0 # Reset counter

            for param_group in self.trainer.optimizer.param_groups:
                param_group['lr'] = self.trainer.initial_lr

            new_T0 = math.ceil(self.trainer.steps_per_epoch * self.trainer.config.scheduler_T0_epoch_fraction)
            if new_T0 < 1: new_T0 = 1

            self.trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.trainer.optimizer,
                T_0=new_T0,
                T_mult=self.trainer.config.scheduler_T_mult,
                eta_min=self.trainer.config.learning_rate * 0.1,
                last_epoch=-1 # Ensure it restarts
            )
            print(f"Test: Plateau detected. Scheduler reset. New T_0={new_T0}") # For test verbosity


    def test_lr_cosine_annealing_and_plateau_restart(self):
        lrs = []

        # Expected T_0 for the first cycle and after restarts
        expected_T0 = math.ceil(self.steps_per_epoch * self.config.scheduler_T0_epoch_fraction)

        # Simulate some training steps
        # Adjust total_simulated_steps based on eval frequency and sequence length to see multiple plateaus.
        val_metric_sequence = [10.0, 9.0, 8.0, # Improving
                               7.9, 7.8, # Still improving slightly
                               7.7, 7.7, # Plateau starts (eval 1, patience 1)
                               7.7, 7.7, # Plateau continues (eval 2, patience 2 -> restart)
                               7.6, # Improvement after restart
                               7.5, 7.4, 7.3, 7.3, 7.3] # another plateau
        num_evals_needed = len(val_metric_sequence)
        total_simulated_steps = num_evals_needed * self.config.eval_every + expected_T0 # Ensure enough steps to see effects past last eval

        current_step_in_cycle = 0 # Relative to current T0 cycle
        eval_step_counter = 0
        eval_idx = 0

        print(f"Starting test: total_simulated_steps={total_simulated_steps}, expected_T0={expected_T0}")

        for step in range(total_simulated_steps):
            # In real training, optimizer.step() is called in train_step, then scheduler.step()
            # For this test, we only care about scheduler.step() effect on LR
            self.trainer.scheduler.step()
            lrs.append(self.trainer.scheduler.get_last_lr()[0])

            # Determine current_step_in_cycle based on actual scheduler state if possible,
            # or by tracking resets. For CosineAnnealingWarmRestarts, self.trainer.scheduler.last_epoch gives steps in current cycle.
            current_step_in_cycle = self.trainer.scheduler.last_epoch +1 # last_epoch is 0-indexed for steps taken

            eval_step_counter += 1
            if eval_step_counter % self.config.eval_every == 0:
                if eval_idx < len(val_metric_sequence):
                    current_val_metric = val_metric_sequence[eval_idx]
                    print(f"Simulating eval at step {step+1} (eval_idx {eval_idx}), val_metric: {current_val_metric}, LR: {lrs[-1]:.6f}")

                    # Store LR before eval to check if it reset
                    lr_before_eval = lrs[-1]
                    self.simulate_evaluation(current_val_metric)
                    lr_after_eval = self.trainer.optimizer.param_groups[0]['lr'] # LR in optimizer after potential reset

                    # If LR was reset by simulate_evaluation, the *next* scheduler.step() will use it.
                    # The current lrs[-1] is from scheduler.step() *before* simulate_evaluation potentially changed optimizer LR.
                    if lr_after_eval == self.trainer.initial_lr and lr_before_eval != self.trainer.initial_lr :
                         print(f"LR reset detected at step {step+1}. Optimizer LR now {lr_after_eval:.6f}")
                         # The *next* lrs entry will reflect this peak.
                    eval_idx +=1

            # --- Basic Checks ---
            self.assertGreaterEqual(lrs[-1], self.config.learning_rate * 0.1 - 1e-7, msg=f"LR {lrs[-1]} too low at step {step+1}") # eta_min check
            self.assertLessEqual(lrs[-1], self.config.learning_rate + 1e-7, msg=f"LR {lrs[-1]} too high at step {step+1}") # initial_lr check

        # For visual inspection:
        # import matplotlib.pyplot as plt
        # plt.plot(lrs)
        # plt.xlabel("Simulated Step"); plt.ylabel("Learning Rate"); plt.title("Simulated LR Schedule")
        # plt.savefig("lr_schedule_test.png"); print("Saved lr_schedule_test.png")

        # --- Specific Assertions ---
        # Check for LR reset points. Eval every 5 steps. Patience 2.
        # Plateau 1: val_metrics[5,6,7] are 7.7, 7.7, 7.7
        # Eval 1 (step 4 in lrs): val_metric[5]=7.7. Patience becomes 1 after this eval if no prior improvement.
        # Eval 2 (step 9 in lrs): val_metric[6]=7.7. Patience becomes 2. Plateau detected. LR reset for step 10.
        # Let's trace:
        # steps 0-3: normal ops
        # step 4 (eval_every=5, 0-indexed): eval with val_metric[5]=7.7. Patience -> 1.
        # steps 5-8: normal ops
        # step 9 (eval_every=5): eval with val_metric[6]=7.7. Patience -> 2. Plateau! Optimizer LR set to initial_lr. Scheduler re-created.
        # LR for step 10 (lrs[9]) should be peak.
        # The LR recorded at lrs[9] is *after* scheduler.step() on step 9.
        # simulate_evaluation happens *after* this LR is recorded for step 9.
        # So, the LR at lrs[10] (for step 11) should be the peak.

        # Let's re-verify indices:
        # Step 0: scheduler.step(), lrs[0]
        # ...
        # Step 4: scheduler.step(), lrs[4]. eval_step_counter = 5. Eval with val_metric_sequence[eval_idx=2]=8.0. (assume some prior best)
        #   simulate_evaluation(8.0). self.trainer.plateau_best_metric_val = 8.0. self.trainer.plateau_patience_counter = 0.
        # Step 5..8
        # Step 9: scheduler.step(), lrs[9]. eval_step_counter = 10. Eval with val_metric_sequence[eval_idx=5]=7.7.
        #   simulate_evaluation(7.7). plateau_best_metric_val=7.7, patience_counter=0 (assuming 7.8 was prev best)
        # Step 14: scheduler.step(), lrs[14]. eval_step_counter = 15. Eval with val_metric_sequence[eval_idx=6]=7.7
        #   simulate_evaluation(7.7). Patience_counter -> 1
        # Step 19: scheduler.step(), lrs[19]. eval_step_counter = 20. Eval with val_metric_sequence[eval_idx=7]=7.7
        #   simulate_evaluation(7.7). Patience_counter -> 2. PLATEAU! Opt LR reset, Scheduler re-init.
        # The LR for step 20 (lrs[19]) is recorded *before* the reset for step 20.
        # The reset affects the LR that will be used for step 20's *parameter update* and recorded as lrs[20].
        self.assertAlmostEqual(lrs[20], self.config.learning_rate, places=6,
                               msg=f"LR should reset after first plateau. Expected peak at lrs[20]. Got {lrs[20]}. Full LRs: {lrs[:25]}")

        # Second plateau: val_metric_sequence[11,12,13] are 7.3, 7.3, 7.3
        # Previous plateau reset LR at step 20. Best val was 7.7.
        # Step 24 (eval_idx=8): val_metric=7.6. best=7.6, patience=0
        # Step 29 (eval_idx=9): val_metric=7.5. best=7.5, patience=0
        # Step 34 (eval_idx=10): val_metric=7.4. best=7.4, patience=0
        # Step 39 (eval_idx=11): val_metric=7.3. best=7.3, patience=0
        # Step 44 (eval_idx=12): val_metric=7.3. patience=1
        # Step 49 (eval_idx=13): val_metric=7.3. patience=2 -> PLATEAU!
        # LR for step 50 (lrs[49]) is before reset. LR for step 50 (lrs[50]) should be peak.
        if total_simulated_steps > 50:
            self.assertAlmostEqual(lrs[50], self.config.learning_rate, places=6,
                                   msg=f"LR should reset after second plateau. Expected peak at lrs[50]. Got {lrs[50]}. Full LRs: {lrs[45:55]}")

if __name__ == '__main__':
    unittest.main()
