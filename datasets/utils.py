from .coco import *
from .soda import *
from .voc import *


def register_common_datasets(voc=True, soda=True, coco=True):
    """
    Registers commonly used datasets in the DataSetCatalog of Detectron2. See respective methods for details.
    Registered datasets:
        - VOC2007: (split = [train, val, trainval, test])
            - VOC_{split}_1-19
            - VOC_{split}_20
            - VOC_{split}_1-10
            - VOC_{split}_11-20
            - VOC_{split}_1-15
            - VOC_{split}_16-20
            - VOC_{split}_11-12
            - VOC_{split}_13-14
            - VOC_{split}_15-16
            - VOC_{split}_17-18
            - VOC_{split}_19-20
            - VOC_{split}_16
            - VOC_{split}_17
            - VOC_{split}_18
            - VOC_{split}_19
            - VOC_finetune10 (A small set with at least 10 images of each class)
        - SODA10M track 3b: (split = [train, val, test])
            - haitain1_{split}
            - haitain2_{split}
            - haitain3_{split}
            - haitain4_{split}
        - Microsoft COCO: (split = [train, val])
            - COCO_{split}2014_task_1
            - COCO_{split}2014_task_2
    """
    if voc:
        register_class_incremental_voc([list(range(19)), [19],
                                        list(range(10)), list(range(10, 20)),
                                        list(range(15)), list(range(15, 20)),
                                        list(range(10, 12)), list(range(12, 14)), list(range(14, 16)), list(range(16, 18)),
                                        list(range(18, 20)), [15], [16], [17], [18]],

                                       ["1-19", "20", "1-10", "11-20", "1-15", "16-20", "11-12", "13-14",
                                        "15-16", "17-18", "19-20", "16", "17", "18", "19"])
        register_voc()
        register_finetune_voc()

    if soda:
        register_track_3b_sets()

    if coco:
        register_ci_coco()
