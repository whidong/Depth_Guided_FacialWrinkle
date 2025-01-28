import torch
import torch.nn.functional as F

def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
    to_onehot_y: bool = True,
    apply_softmax: bool = True,
    include_background: bool = True,
    n_classes: int = 2,
) -> torch.Tensor:
    """
    Soft Dice Loss 구현 (멀티 클래스 가능)
    로그나이즈 코드(DiceLoss)처럼 to_onehot, softmax, background 채널 포함 여부를 지원.

    Args:
        logits (Tensor): 모델 출력 (B, C, H, W, ...) 형태.
        targets (Tensor): 정답 라벨 (B, H, W, ...) 
                          - to_onehot_y=True라면 정수 클래스 레이블
                          - to_onehot_y=False라면 이미 one-hot 형태 or 다채널 가능
        smooth (float): 분모가 0이 되는걸 방지하는 smoothing 계수
        to_onehot_y (bool): targets를 one-hot 변환할지 여부
        apply_softmax (bool): logits에 softmax 적용할지 여부
        include_background (bool): True면 채널 0(배경)도 dice 계산에 포함
        n_classes (int): 클래스 수 (one-hot 변환, softmax 용)

    Returns:
        dice_loss (Tensor): 스칼라 텐서 (0 이상)
    """
    # 1) softmax 적용 (원한다면)
    if apply_softmax:
        probs = F.softmax(logits, dim=1)
    else:
        probs = logits

    # 2) one-hot 변환 (targets가 정수 레이블이라면)
    if to_onehot_y:
        # targets: (B, H, W) -> (B, H, W, n_classes) -> (B, n_classes, H, W)
        targets_onehot = F.one_hot(targets, num_classes=n_classes).permute(0, 3, 1, 2).float()
    else:
        # 이미 one-hot 형태라면 바로 사용
        # 혹은 (B, C, H, W) 형태라면 그대로
        targets_onehot = targets.float()

    # 3) background 채널 제외 여부
    start_channel = 0 if include_background else 1
    probs = probs[:, start_channel:, ...]            # (B, C', ...)
    targets_onehot = targets_onehot[:, start_channel:, ...]  # (B, C', ...)

    # 4) 채널/공간 차원을 기준으로 Dice 계산
    # intersection: (B, C', H, W) → 모든 공간합
    intersection = (probs * targets_onehot).sum(dim=[0, 2, 3])  # 채널별 합
    denom = probs.sum(dim=[0, 2, 3]) + targets_onehot.sum(dim=[0, 2, 3])

    dice_per_channel = (2.0 * intersection + smooth) / (denom + smooth)
    dice_score = dice_per_channel.mean()    # 모든 채널 평균
    dice_loss = 1.0 - dice_score

    return dice_loss
