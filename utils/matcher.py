from scipy.optimize import linear_sum_assignment
import torch
from utils.losses import diou
from utils.functions import box_cxcywh_to_xyxy
from utils import config

@torch.no_grad()
def matcher(predicted, targets):
  device = torch.device("cpu")

  matchings = []
  for image_class_predicted_probability_distributions, \
      image_bbox_predictions, \
      image_class_target_labels, \
      image_bbox_targets in \
    zip(
      predicted['class_predictions'].to(device), 
      predicted['bbox_predictions'].to(device),
      [v['labels'].to(device) for v in targets],
      [v['boxes'].to(device) for v in targets]
    ):
      if image_class_target_labels.shape[0] > 0:
        cost_class = -image_class_predicted_probability_distributions[:, image_class_target_labels]
        cost_bbox = torch.cdist(image_bbox_predictions, image_bbox_targets, p=1)
        cost_giou = -diou(box_cxcywh_to_xyxy(image_bbox_predictions), box_cxcywh_to_xyxy(image_bbox_targets))

        cost_class = cost_class.cpu().detach()
        cost_bbox = cost_bbox.cpu().detach()
        cost_giou = cost_giou.cpu().detach()

        config.tb.add_histogram("cost_class", cost_class, config.epoch)
        config.tb.add_histogram("cost_bbox", cost_bbox, config.epoch)
        config.tb.add_histogram("cost_giou", cost_giou, config.epoch)

        cost_matrix = cost_class + cost_bbox + cost_giou
        cost_matrix = cost_matrix.cpu().detach()

        solution_row_ind, solution_col_ind = linear_sum_assignment(cost_matrix)

        matchings.append(
          (torch.as_tensor(solution_row_ind, dtype=torch.int64),
           torch.as_tensor(solution_col_ind, dtype=torch.int64))
        )
      else:
        matchings.append(None)

  return matchings