import copy

import numpy as np
import torch
from detectron2.data import DatasetMapper
import detectron2.data.detection_utils as detection_utils
import detectron2.data.transforms as T


class InferenceDecorator:
    """
    Will decorate the inference call of the model. If rpn_out is set, all outputs of the rpn will be logged to this
    file. If roi_out is set, the logit output of the roi_heads on the gt boxes will be logged.
    """
    def __init__(self, func, rpn_out=None, roi_out=None, nclasses=20):
        self.func = func
        self.model = func.__self__
        self.rpn_out = rpn_out
        self.roi_out = roi_out
        self.nclasses = nclasses

        if roi_out is not None:
            self.features = None
            self.register_backbone_hook()

        if rpn_out is not None:
            self.proposals = None
            self.register_rpn_hook()

        self.init_log_file()

    def register_backbone_hook(self):
        def backbone_hook(module, in_args, out_args):
            self.features = out_args
        self.model.backbone.register_forward_hook(backbone_hook)

    def register_rpn_hook(self):
        def rpn_eval_hook(module, in_args, out_args):
            self.proposals = out_args[0]
        self.model.proposal_generator.register_forward_hook(rpn_eval_hook)

    def __call__(self, batched_inputs, **kwargs):

        results = self.func(batched_inputs, **kwargs)
        gt_instances = [x["instances"].to(self.model.device) for x in batched_inputs]

        if self.roi_out is not None:

            for gt_i in gt_instances:
                gt_i.proposal_boxes = gt_i.gt_boxes

            logits, _ = self.model.roi_heads.get_soft_output(self.features, gt_instances)
            self.log_logits(batched_inputs, logits)

        if self.rpn_out is not None:
            self.log_proposals(batched_inputs, gt_instances)

        return results

    def init_log_file(self):

        if self.roi_out is not None:
            header = 'image_id,class,x1,y1,x2,y2,'
            header += ''.join(f'class {i},' for i in range(self.nclasses))
            header += 'background\n'

            with open(self.roi_out, 'w') as file:
                file.write(header)

        if self.rpn_out is not None:
            header = 'image_id,width,height,score,x1,y1,x2,y2\n'
            with open(self.rpn_out, 'w') as file:
                file.write(header)

    def log_logits(self, batched_inputs, logits):
        if len(batched_inputs) > 1:
            raise NotImplementedError("Logging only implemented for one image at once. "
                                      "Have to look into how logits are structured.")

        fmt_boxes = ('{:.3f},' * 4)[:-1]
        fmt_scores = ('{:.3f},' * (self.nclasses + 1))[:-1]

        img_id = batched_inputs[0]['image_id']
        instances = batched_inputs[0]['instances']

        with open(self.roi_out, 'a') as f:
            for box, cls, pred_logits in zip(instances.gt_boxes, instances.gt_classes, logits):
                f.write(f'{img_id},{cls},{fmt_boxes.format(*box)},{fmt_scores.format(*pred_logits)}\n')

    def log_proposals(self, batched_inputs, gt_instances):
        fmt_boxes = ('{:.3f},' * 4)[:-1]
        with open(self.rpn_out, 'a') as f:
            for img, gts, props in zip(batched_inputs, gt_instances, self.proposals):
                img_id = img['image_id']
                height, width = gts.image_size
                for logit, anchor in zip(props.objectness_logits, props.proposal_boxes.tensor):
                    f.write(f"{img_id},{width},{height},{logit:.3f},{fmt_boxes.format(*anchor)}\n")


class DatasetMapperWithTestAnnotations(DatasetMapper):
    """
    This overrides the default dataset mapper to keep the annotations at inference, which is necessary for some
    logging capabilites we want. Only used for boxes for now, all segmentation code etc. is removed.
    """
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = detection_utils.read_image(dataset_dict["file_name"], format=self.image_format)
        detection_utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore, it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict
