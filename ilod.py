import random
from typing import List, Dict, Tuple, Sequence
from copy import deepcopy

import torch
from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY, ROI_HEADS_REGISTRY
from detectron2.config import configurable
from detectron2.modeling.proposal_generator.proposal_utils import find_top_rpn_proposals
from detectron2.modeling.roi_heads import Res5ROIHeads
from detectron2.modeling.proposal_generator import RPN, PROPOSAL_GENERATOR_REGISTRY
from detectron2.structures import Boxes, Instances, pairwise_iou
import torch.nn.functional as F

"""
Options in the cfg files:

ILOD:
  DISTILLATION: General ILOD flag, if False regular FasterRCNN is used (albeit it slower than normal, no idea
  why). If this is False, all other options are obsolete. Default: False
  LOAD_TEACHER: If the weights that are in the config file contain weights for the teacher and you want
  to load those, this should be True. This is for instance useful when restarting training. When loading
  weights because a new task is started, usually, you want to reinitalize the teacher and therefore this
  should be False. Default: False
  FEATURE_LAM: Scaling parameter for the feature distillation loss. Default: 1.0
  RPN_LAM: Scaling parameter for the RPN distillation loss. Default: 1.0
  ROI_LAM: Scaling parameter for the ROI-head distillation loss. Default: 1.0
  HUBER: If true Huber loss is used, else MSE in the ROI distillation loss
  TEACHER_CORR: Correct the teacher model region proposals based on the current gt's (only used in distillation losses) 
"""


@PROPOSAL_GENERATOR_REGISTRY.register()
class ILODRPN(RPN):
    """
    This is only required because we need the raw RPN predictions and by
    default they are returned sorted. By overwriting predict_proposals we can store them in class members.
    """

    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pred_proposals = None
        self.pred_objectness_logits = None

    def predict_proposals(self, anchors: List[Boxes], pred_objectness_logits: List[torch.Tensor],
                          pred_anchor_deltas: List[torch.Tensor], image_sizes: List[Tuple[int, int]]):
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            # Store the raw predictions, we need to access these later to calculate a loss
            # Could also be done with a hook but this is cleaner.
            self.pred_proposals = pred_anchor_deltas
            self.pred_objectness_logits = pred_objectness_logits
            return find_top_rpn_proposals(pred_proposals, pred_objectness_logits, image_sizes,
                                          self.nms_thresh, self.pre_nms_topk[self.training],
                                          self.post_nms_topk[self.training], self.min_box_size, self.training)


@ROI_HEADS_REGISTRY.register()
class ILODRoiHead(Res5ROIHeads):
    """
    ILOD Roi Head, we need the soft predicitions of the head to calculate the loss, which requires
    a new method.
    """

    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_soft_output(self, features, proposals):
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))
        return predictions


@META_ARCH_REGISTRY.register()
class ILODRCNN(GeneralizedRCNN):
    """
    ILOD model based on a FasterRNN structure. The model keeps a reference to its
    initalization, which is referred to as the teacher model.
    """

    @configurable
    def __init__(self, use_teacher: bool = True, load_teacher: bool = False, feature_lam: float = 1.0,
                 rpn_lam: float = 1.0, roi_lam: float = 1.0, teacher_corr: bool = True, huber: bool = True, **kwargs):
        """
        :param use_teacher: if True, the ILOD method is used. (default = True)
        :param load_teacher: If true, teacher weights will try to be loaded from the given weights. This will give
        errors if those weights aren't present. Should only be True if training is resumed, not if a new task is
        started. (default = False)
        :param feature_lam: scalor for the feature distillation loss (default = 1.0)
        :param rpn_lam: scalor for the rpn distillation loss (default = 1.0)
        :param roi_lam: scalor for the roi distillation loss (default = 1.0)
        :param teacher_corr: (dis/en)ables teacher correction in the ROI-loss.
        :param huber: If True, the MSE distillation loss in the ROI-heads changes gradually to a Huberloss.
        :param kwargs: kwargs for GeneralizedRCNN
        """
        super().__init__(**kwargs)
        self.use_teacher = use_teacher

        self.feature_lam = feature_lam
        self.rpn_lam = rpn_lam
        self.roi_lam = roi_lam

        self.huber = huber
        self.delta_calc = DeltaCalc()  # Keeps track of delta parameter of Huber loss.

        self.teacher_corr = teacher_corr

        # Only here if we have different weights for the teacher than the student. If we want to use the same weights,
        # leave None here and initialize in first forward. This is because the old weights are only loaded after the
        # model's init method has finished.
        self.teacher = deepcopy(self) if self.use_teacher and load_teacher else None

        if self.teacher is not None:
            self.turn_off_teacher_grad()

    @classmethod
    def from_config(cls, cfg):
        base_kwargs = super().from_config(cfg)
        return {
            "use_teacher": cfg.ILOD.DISTILLATION,
            "load_teacher": cfg.ILOD.LOAD_TEACHER,
            "feature_lam": cfg.ILOD.FEATURE_LAM,
            "rpn_lam": cfg.ILOD.RPN_LAM,
            "roi_lam": cfg.ILOD.ROI_LAM,
            "huber": cfg.ILOD.HUBER,
            'teacher_corr': cfg.ILOD.TEACHER_CORR,
            **base_kwargs
        }

    def turn_off_teacher_grad(self):
        for param in self.teacher.parameters():
            param.requires_grad = False

    def teacher_init(self):
        """
        This will create a deepcopy of self in teacher. Then, the gradients of those weights are turned off.
        Setting to eval mode will also change the number of proposals etc., which shouldn't happen.
        """
        self.teacher = deepcopy(self)
        self.turn_off_teacher_grad()

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Faster-ILOD forward call.
        """

        if self.use_teacher and self.teacher is None:
            self.teacher_init()

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.use_teacher:

            teacher_features = self.teacher.backbone(images.tensor)
            if self.feature_lam > 0:
                feature_distil_loss = calc_feature_distil_loss(teacher_features, features)
                losses["feature_distill_loss"] = self.feature_lam * feature_distil_loss

            teacher_proposals, _ = self.teacher.proposal_generator(images, teacher_features, gt_instances)

            if self.rpn_lam > 0:
                rpn_distil_loss = calc_rpn_distil_loss(self.teacher.proposal_generator.pred_proposals,
                                                       self.teacher.proposal_generator.pred_objectness_logits,
                                                       self.proposal_generator.pred_proposals,
                                                       self.proposal_generator.pred_objectness_logits,
                                                       bbox_threshold=0.1)
                losses["rpn_distill_loss"] = self.rpn_lam * rpn_distil_loss

            if self.roi_lam > 0:

                soft_teacher, soft_student, soft_proposals = \
                    self.get_soft_proposals(teacher_features, teacher_proposals, features, gt_instances)
                roi_distil_loss = calc_roi_distillation_losses(soft_teacher, soft_student, self.huber,
                                                               delta=self.delta_calc.calc_delta())
                losses["roi_distill_loss"] = self.roi_lam * roi_distil_loss

        return losses

    def get_soft_proposals(self, source_features: Dict[str, torch.Tensor], source_proposals: Sequence[Instances],
                           target_features: Dict[str, torch.Tensor], gt_instances: List[Instances]) \
            -> Tuple[Tuple[torch.Tensor, Boxes], Tuple[torch.Tensor, Boxes], List[Instances]]:
        """
        Select 64 out of the 128 highest scoring proposals of the teacher, feed those the ROI heads
        of the student.
        :param source_features: Features of teacher network
        :param source_proposals: Proposals of the teacher network
        :param target_features: Features of the student network
        :param gt_instances: GT instances. If proposals have high IOU with current GT, they aren't used in the
        soft porposals.
        :return: Two tuples with soft scores and boxes for student and teacher, and the proposals
        """
        proposals = []
        for source_props, gt in zip(source_proposals, gt_instances):
            scores, scores_idx = source_props.get("objectness_logits").sort(descending=True)
            img_proposals = Instances.cat([source_props[idx.item()] for idx in scores_idx])

            if self.teacher_corr:
                ious = pairwise_iou(gt.gt_boxes, img_proposals.proposal_boxes)
                gt_matches = torch.max(ious, dim=0).values < 0.5
                img_proposals = [img_proposals[i] for i, gtm in enumerate(gt_matches) if gtm]

            nb_proposals = len(img_proposals)

            if nb_proposals < 64:
                random_proposals_idx = range(0, nb_proposals)
            elif nb_proposals < 128:
                random_proposals_idx = random.sample(range(0, nb_proposals), 64)
            else:
                random_proposals_idx = random.sample(range(0, 128), 64)

            proposals.append(Instances.cat([img_proposals[idx] for idx in random_proposals_idx]))

        teacher_soft = self.teacher.roi_heads.get_soft_output(source_features, proposals)
        student_soft = self.roi_heads.get_soft_output(target_features, proposals)

        return teacher_soft, student_soft, proposals


def calc_feature_distil_loss(teacher_features: Dict[str, torch.Tensor],
                             student_features: Dict[str, torch.Tensor]) -> float:
    """
    Calcualtes the feature distillation loss, as the difference between the two losses.
    First the features are normalized to have zero mean.
    :param teacher_features: features of the teacher model
    :param student_features: features of the student model
    :return: Feature distillation loss
    """
    # Currently all features are averaged over all features together, rather than per feature.
    final_feature_distillation_loss = []
    assert teacher_features.keys() == student_features.keys()

    for key in student_features.keys():
        teacher_feature = teacher_features[key]
        student_feature = student_features[key]
        normalized_teacher_feature = teacher_feature - torch.mean(teacher_feature)  # normalize features
        normalized_student_feature = student_feature - torch.mean(student_feature)

        feature_difference = normalized_teacher_feature - normalized_student_feature
        feature_size = feature_difference.size()

        # (This is to delete al where the f_st is higher than the f_te)
        mask = torch.zeros(feature_size, device=teacher_feature.device)
        feature_distillation_loss = torch.max(feature_difference, mask)

        final_feature_distillation_loss.append(torch.mean(feature_distillation_loss))
        del mask
        torch.cuda.empty_cache()  # Release unoccupied memory

    final_feature_distillation_loss = sum(final_feature_distillation_loss)
    return final_feature_distillation_loss


def calc_rpn_distil_loss(teacher_regression, teacher_scores, student_regression, student_scores, bbox_threshold=0.1):
    """
    Calcultes the RPN distillation loss. It takes all regression and objectness score outputs of both the teacher
    and student network. A prediction is only included if the teacher score is higher than the student, and its
    regression only if the teachers score is `bbox_treshold` higher than the students score. Both losses are MSE.
    All parameters should be ordered identically. Note that this is not by default the output of Detectron2 RPN's.
    You should use ILODRPN to get access to the unsorted scores and boxes.

    :param teacher_regression:
    :param teacher_scores:
    :param student_regression:
    :param student_scores:
    :param bbox_threshold: Treshold to include the bbox regression for a given prediction.
    :return: sum of the losses, divided by the number of proposals.
    """
    teacher_scores = teacher_scores[0].view(-1)
    teacher_regression = teacher_regression[0].view(-1, 4)
    student_scores = student_scores[0].view(-1)
    student_regression = student_regression[0].view(-1, 4)

    assert len(teacher_scores) == len(student_scores), \
        f"Nb of teacher proposals{len(teacher_scores)} doesn't equal student propsals {len(student_scores)}"

    nb_proposals = len(teacher_scores)

    mask = torch.zeros(nb_proposals, dtype=torch.bool)
    mask[teacher_scores > student_scores] = 1
    score_diff = student_scores[mask] - teacher_scores[mask]
    score_loss = torch.dot(score_diff, score_diff) if not torch.all(mask == 0) else torch.tensor(0.0)

    mask = torch.zeros(nb_proposals, dtype=torch.bool)
    mask[teacher_scores >= (student_scores + bbox_threshold)] = 1
    regres_diff = student_regression[mask] - teacher_regression[mask]
    regres_diff = regres_diff.view(-1)
    regres_loss = torch.dot(regres_diff, regres_diff) if not torch.all(mask == 0) else torch.tensor(0.0)

    return 1 / nb_proposals * (regres_loss + score_loss)


def calc_roi_distillation_losses(soft_teacher, soft_student, huber: bool = True, delta: float = 4):
    """
    Tihs calculates the ROI-distillation loss given soft scores and boxes calculated by both the teacher and
    student network on the same proposals. They should be ordered identically, since there's no way to check
    here. The scores are first normalized across all classes. (This reduces the effect of having larger overall scores
    in the student network.).
    :param soft_teacher: scores and boxes from the teacher
    :param soft_student: scores and boxes from the student
    :param huber: If True, Huber loss will be used. Else MSE is used.
    :param delta: Delta parameter to use. Only effective if huber is True.
    :return: loss
    """
    soft_teach_scores, soft_teach_boxes = soft_teacher
    soft_stud_scores, soft_stud_boxes = soft_student

    soft_teach_scores = soft_teach_scores - torch.mean(soft_teach_scores, dim=1, keepdim=True)
    soft_stud_scores = soft_stud_scores - torch.mean(soft_stud_scores, dim=1, keepdim=True)

    if not huber:
        score_loss = F.mse_loss(soft_stud_scores, soft_teach_scores)
        box_loss = F.mse_loss(soft_stud_boxes, soft_teach_boxes)
    else:
        # Times 2 to match the MSE exactly for losses < delta
        score_loss = 2 * F.huber_loss(soft_stud_scores, soft_teach_scores, delta=delta)
        box_loss = 2 * F.huber_loss(soft_stud_boxes, soft_teach_boxes, delta=delta)

    return score_loss + box_loss


class DeltaCalc:

    def __init__(self, delta_max: float = 4, delta_min: float = .5, momentum: float = 0.9995,
                 warm_up: int = 100):
        """
        Calculates and stores an exponentionally decaying parameter. Each calculation (calc_delta) delta is
        exponentionally decayed. Delta is calculated as: momentum * (d - d_min) + d_min and initalized at d_max.
        :param delta_max: Maximum value of delta
        :param delta_min: Minimum value of delta. Delta will asymptotically approach this.
        :param momentum: Mometenum to decay with.
        :param warm_up: Number of warm-up iterations before decaying starts.
        """

        self.delta_max = delta_max
        self.delta_min = delta_min
        self.momentum = momentum

        self.warm_up_iter = warm_up
        self.current_iter = 0
        self.delta = self.delta_max

    def calc_delta(self) -> float:
        """
        Increment the iteration count, decay delta and return it.
        :return: decayed delta
        """
        self.current_iter += 1

        if self.current_iter < self.warm_up_iter:
            return self.delta
        else:
            self.delta = self.momentum * (self.delta - self.delta_min) + self.delta_min
            return self.delta
