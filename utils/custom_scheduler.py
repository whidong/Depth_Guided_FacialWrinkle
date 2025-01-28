import math
import torch.optim as optim

class CustomCosineAnnealingWarmRestarts(optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, eta_max=0.0001, decay_factor=0.9, start_epoch=0, last_epoch=-1):
        self.eta_max = eta_max
        self.decay_factor = decay_factor
        self.current_cycle = 0
        self.start_epoch = start_epoch  # 스케줄링 시작 epoch
        self.cycle_offset = 0  # warm restart 때마다 T_total을 더해 줄 변수
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
    def get_lr(self):
        """
        현재 학습률 계산 (start_epoch 이전에는 기본 학습률 유지)
        """
        if self.last_epoch < self.start_epoch:
            # start_epoch 이전에는 기본 학습률 반환
            return [base_lr for base_lr in self.base_lrs]

        T_cur = self.T_cur
        T_total = self.T_0 * (self.T_mult ** self.current_cycle)  # 현재 주기의 전체 길이

        # Cosine Annealing 수식에 따라 학습률 계산
        return [
            self.eta_min + (self.eta_max - self.eta_min) *
            (1 + math.cos(math.pi * T_cur / T_total)) / 2
            for _ in self.base_lrs
        ]

    def step(self, epoch=None):
        """
        학습률 및 주기 갱신
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            self.last_epoch = epoch

        if epoch < self.start_epoch:
            # start_epoch 이전에는 아무런 동작을 수행하지 않음
            self.last_epoch = epoch
            return
        
        effective_epoch = epoch - self.start_epoch

        # 주기 갱신 로직
        T_total = self.T_0 * (self.T_mult ** self.current_cycle)
        
        self.T_cur = effective_epoch - self.cycle_offset
        print(f"[DEBUG step] epoch={epoch}, effective_epoch={effective_epoch}, T_cur={self.T_cur}, T_total={T_total}, current_cycle={self.current_cycle}, eta_max={self.eta_max}")

        if self.T_cur >= T_total:
            self.current_cycle += 1
            self.eta_max *= self.decay_factor
            self.cycle_offset += T_total  # 지나간 주기 길이를 누적
            self.T_cur = 0
            print(f"[DEBUG step] Warm Restart triggered: current_cycle={self.current_cycle}, new eta_max={self.eta_max}")

        # 부모 클래스의 step 호출
        super().step(epoch)

        # 학습률 갱신
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        print(f"[DEBUG step] After step: last_epoch={self.last_epoch}, T_cur={self.T_cur}, current_cycle={self.current_cycle}, _last_lr={self._last_lr}")
