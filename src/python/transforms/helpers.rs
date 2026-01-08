// Helper functions for transform wrappers

use crate::sampling::RandomImageNode;
use crate::sampling::Dist;

/// Helper function to wrap a node in Maybe if p is not always-active
#[cfg(feature = "python")]
pub(crate) fn maybe_wrap(node: RandomImageNode, p: Dist) -> RandomImageNode {
    // If p is Constant(1.0), just return the node directly
    // Otherwise, wrap in Maybe
    if matches!(p, Dist::Constant(1.0)) {
        node
    } else {
        RandomImageNode::Maybe {
            child: Box::new(node),
            p,
        }
    }
}
