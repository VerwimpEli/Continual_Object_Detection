import logging
import os
import argparse
from datetime import datetime

from detectron2.model_zoo import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger

# Import ilod to register the modules.
import ilod
from log_utils import DatasetMapperWithTestAnnotations, InferenceDecorator
import datasets as ds

# TODO: add multi gpu support


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="yaml config file")
    parser.add_argument('--outdir', help="Base output dir")
    parser.add_argument('--no_cuda', action='store_true', help="Run on cpu")
    parser.add_argument('--test_only', action='store_true', help="Don't train, only test")
    parser.add_argument('--dump_rpn', help="Dump RPN proposals at inference time in output dir")
    parser.add_argument('--dump_roi', help="Dump ROI soft predictions in output dir, during training. ")
    parser.add_argument('--load_last', action='store_true',
                        help='load last trained model, source folder is in last_model.txt')
    args = parser.parse_args()

    ds.register_common_datasets(voc=True, soda=True, coco=False)
    cfg = setup(args)

    if not args.test_only:
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        # Can only load last model if trained. Else the weights should be in the cfg.
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    setup_detectron_logger(args)
    predictor = DefaultPredictor(cfg)

    # Decorate inference call to log rpn and roi outputs at inference.
    args.dump_roi = f"{cfg.OUTPUT_DIR}/{args.dump_roi}" if args.dump_roi is not None else None
    args.dump_rpn = f"{cfg.OUTPUT_DIR}/{args.dump_rpn}" if args.dump_rpn is not None else None

    predictor.model.inference = InferenceDecorator(predictor.model.inference,
                                                   roi_out=args.dump_roi,
                                                   rpn_out=args.dump_rpn)

    for test_dataset in cfg.DATASETS.TEST:
        evaluator = COCOEvaluator(test_dataset, output_dir=f"{cfg.OUTPUT_DIR}/{test_dataset}")
        val_loader = build_detection_test_loader(cfg, test_dataset,
                                                 mapper=DatasetMapperWithTestAnnotations(cfg, False))
        print(inference_on_dataset(predictor.model, val_loader, evaluator))

    # Write path of last model to a file. This can be used the next time this script is run.
    with open(os.path.join(args.outdir, 'last_model.txt'), 'w') as f:
        f.write(f"{cfg.OUTPUT_DIR}\n")


def setup(args):
    """
    Sets up config file.
    """

    cfg = get_cfg()
    add_ilod_config(cfg)

    # Loads basic config file and then merges with args.config
    cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"))
    cfg.merge_from_file(args.config)
    cfg.MODEL.DEVICE = "cpu" if args.no_cuda else cfg.MODEL.DEVICE

    # Loads model at the path in last_model.txt in the current output dir.
    if args.load_last:
        with open(os.path.join(args.outdir, 'last_model.txt'), 'r') as f:
            last_model = f.read().strip()
        cfg.MODEL.WEIGHTS = f"{last_model}/model_final.pth"

    # Makes a new sub-folder in the current output dir to store results of this run.
    if args.outdir is not None:
        time_stamp = datetime.now().strftime('%H%M%S')
        cfg.OUTPUT_DIR = f"{args.outdir}/output_{time_stamp}"
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def setup_detectron_logger(args):
    # If there's no training, set up logger.
    if args.test_only:
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()


def add_ilod_config(cfg):
    _C = cfg
    _C.ILOD = CfgNode()
    _C.ILOD.DISTILLATION = False
    _C.ILOD.LOAD_TEACHER = False
    _C.ILOD.FEATURE_LAM = 1.0
    _C.ILOD.RPN_LAM = 1.0
    _C.ILOD.ROI_LAM = 1.0
    _C.ILOD.HUBER = True
    _C.ILOD.TEACHER_CORR = True


if __name__ == '__main__':
    main()
