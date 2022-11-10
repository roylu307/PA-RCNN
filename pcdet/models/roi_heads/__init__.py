from .roi_head_template import RoIHeadTemplate

from .parcnn_head_pv import PA_RCNNHead_PV
from .parcnn_head_v import PA_RCNNHead_V


__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PA_RCNNHead_PV': PA_RCNNHead_PV,
    'PA_RCNNHead_V': PA_RCNNHead_V,
}
