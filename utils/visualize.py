from utils.functions import box_xyxy_to_cxcywh, uniform_shape_masks
from matplotlib import patches

def visualize(f, axes_images, axes_masks, axes_pred, axes_target_masks, predicted, inputs, targets):
  # class_predictions = predicted['class_predictions']
  # bbox_predictions = predicted['bbox_predictions'] 
  segmentation_masks = predicted['segmentation_masks'].cpu()

  batch_tensor_bhwc = inputs.rearrange("bs c h w -> bs h w c")

  for im, ax in zip(batch_tensor_bhwc.data, axes_images):
    im = im - im.min()
    im /= im.max()
    # im *= 255
    # print(im.min(), im.max())
    ax.imshow(im)

  for mask, ax in zip(batch_tensor_bhwc.mask, axes_masks):
    ax.imshow(mask.float().squeeze(-1))

  for mask, ax in zip(segmentation_masks.detach(), axes_pred):
    ax.imshow(mask[0:3].permute(1,2,0))

  target_masks = uniform_shape_masks([t['masks'] for t in targets]).float()
  target_masks_count = [t['masks'].shape[0] for t in targets]
  for t, t_count, ax in zip(target_masks, target_masks_count, axes_target_masks):
    ax.imshow(t[0:3].permute(1,2,0))


  f.suptitle("bs: {}, height: {}, width: {}, channels: {}".format(*batch_tensor_bhwc.shape));

  return f


def draw_box(ax, box, linewidth=1, edgecolor='b', facecolor='none', text=None, label=None, **kwargs):
  cx,cy,w,h = box_xyxy_to_cxcywh(box).unbind(-1)
  anchor_point = (cx-w/2, cy-h/2)
  rect = patches.Rectangle(anchor_point, w, h, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor, label=label, **kwargs)
  ax.add_patch(rect)
  if text is not None:
    ax.text(cx,cy,text)
  ax.scatter(cx, cy, c=edgecolor, s=linewidth)