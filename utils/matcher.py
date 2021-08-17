from utils.losses import pairwise_strange_loss
from scipy.optimize import linear_sum_assignment
import torch
from utils.functions import box_cxcywh_to_xyxy
from utils import config

@torch.no_grad()
def matcher(predicted, targets):
  """ Uses pairwise_strange_loss. """
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
        cost_bbox = pairwise_strange_loss(image_bbox_predictions, image_bbox_targets)

        cost_class = cost_class.cpu().detach()
        cost_bbox = cost_bbox.cpu().detach()

        cost_matrix = cost_class + cost_bbox
        cost_matrix = cost_matrix.cpu().detach()

        solution_row_ind, solution_col_ind = linear_sum_assignment(cost_matrix)

        matchings.append(
          (torch.as_tensor(solution_row_ind, dtype=torch.int64),
           torch.as_tensor(solution_col_ind, dtype=torch.int64))
        )
      else:
        matchings.append(None)

  return matchings