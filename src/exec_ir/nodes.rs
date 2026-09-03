// Execution IR nodes
//
// Defines ExecNode and ExecPlan types for the optimized execution representation.

use crate::core::{BarrierImage, FusableImage};
use crate::exec_ir::optimizer::BlockStats;
use crate::sampled_ir::SampledImageOp;
use std::fmt;

///// Fast-path kernel function pointer (legacy trait object closure)
pub type FastKernel = Box<dyn Fn(&mut FusableImage) -> Option<BarrierImage> + Send + Sync>;

/// Concrete execution kernel without trait objects or dynamic dispatch
#[derive(Debug, Clone)]
pub enum KernelKind {
    /// Fused LUT: applies 3-channel or 1-channel precomputed lookup tables
    FusedLut {
        luts_3c: Option<Box<[[u8; 256]; 3]>>,
        lut_1c: [u8; 256],
    },
    /// Pure geometric D4 transform (Flips, Rot90, Transpose)
    Geometric(crate::transforms::Orientation),
    /// Resize fused with trailing LUTs
    ResizeWithLut {
        resize: crate::transforms::Resize,
        luts_3c: Option<Box<[[u8; 256]; 3]>>,
        lut_1c: [u8; 256],
    },
    /// Fused LUT followed by ToGray
    LutToGray {
        luts_3c: Box<[[u8; 256]; 3]>,
    },
    /// Equalize with trailing LUTs
    EqualizeWithLut {
        luts_3c: Option<Box<[[u8; 256]; 3]>>,
        lut_1c: [u8; 256],
    },
    /// AutoContrast with trailing LUTs
    AutoContrastWithLut {
        cutoff: f32,
        luts_3c: Option<Box<[[u8; 256]; 3]>>,
        lut_1c: [u8; 256],
    },
    /// Fused Color Matrix
    FusedMatrix(crate::transforms::FusedMatrix),
    /// Single transform execution
    Single(SampledImageOp),
    /// Barrier transform
    Barrier(SampledImageOp),
}

impl KernelKind {
    /// Execute this kernel on an image using static enum dispatch
    #[inline]
    pub fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        match self {
            KernelKind::FusedLut { luts_3c, lut_1c } => {
                if let Some(ref luts) = luts_3c {
                    if image.channels == 3 {
                        crate::transforms::runtime::lut::LutExecutor::apply_rgb_luts(image, luts);
                    } else {
                        crate::transforms::runtime::lut::LutExecutor::apply(image, lut_1c);
                    }
                } else {
                    crate::transforms::runtime::lut::LutExecutor::apply(image, lut_1c);
                }
                None
            }
            KernelKind::Geometric(orientation) => {
                let k = crate::transforms::StructuralKernel::new(*orientation);
                crate::core::Executable::execute(&k, image)
            }
            KernelKind::ResizeWithLut { resize, luts_3c, lut_1c } => {
                Some(resize.apply_with_lut(image, luts_3c.as_deref(), lut_1c))
            }
            KernelKind::LutToGray { luts_3c } => {
                crate::transforms::ToGray::apply_with_lut(image, luts_3c)
            }
            KernelKind::EqualizeWithLut { luts_3c, lut_1c } => {
                if let Some(eq_luts) = crate::transforms::Equalize::build_luts_from_image(image) {
                    if let Some(ref post_luts) = luts_3c {
                        let mut composed = [[0u8; 256]; 3];
                        for c in 0..3 {
                            for i in 0..256 {
                                composed[c][i] = post_luts[c][eq_luts[c][i] as usize];
                            }
                        }
                        if image.channels == 3 {
                            crate::transforms::runtime::lut::LutExecutor::apply_rgb_luts(image, &composed);
                        } else {
                            let mut composed_1c = [0u8; 256];
                            for i in 0..256 {
                                composed_1c[i] = lut_1c[eq_luts[0][i] as usize];
                            }
                            crate::transforms::runtime::lut::LutExecutor::apply(image, &composed_1c);
                        }
                    } else {
                        let mut composed = [[0u8; 256]; 3];
                        for c in 0..3 {
                            for i in 0..256 {
                                composed[c][i] = lut_1c[eq_luts[c][i] as usize];
                            }
                        }
                        if image.channels == 3 {
                            crate::transforms::runtime::lut::LutExecutor::apply_rgb_luts(image, &composed);
                        } else {
                            let mut composed_1c = [0u8; 256];
                            for i in 0..256 {
                                composed_1c[i] = lut_1c[eq_luts[0][i] as usize];
                            }
                            crate::transforms::runtime::lut::LutExecutor::apply(image, &composed_1c);
                        }
                    }
                } else {
                    let _ = crate::core::Executable::execute(&crate::transforms::Equalize::new(), image);
                    if let Some(ref luts) = luts_3c {
                        if image.channels == 3 {
                            crate::transforms::runtime::lut::LutExecutor::apply_rgb_luts(image, luts);
                        } else {
                            crate::transforms::runtime::lut::LutExecutor::apply(image, lut_1c);
                        }
                    } else {
                        crate::transforms::runtime::lut::LutExecutor::apply(image, lut_1c);
                    }
                }
                None
            }
            KernelKind::AutoContrastWithLut { cutoff, luts_3c, lut_1c } => {
                let auto_lut = crate::transforms::AutoContrast::new(*cutoff).build_lut_from_image(image);
                if let Some(ref post_luts) = luts_3c {
                    let mut composed = [[0u8; 256]; 3];
                    for c in 0..3 {
                        for i in 0..256 {
                            composed[c][i] = post_luts[c][auto_lut[i] as usize];
                        }
                    }
                    if image.channels == 3 {
                        crate::transforms::runtime::lut::LutExecutor::apply_rgb_luts(image, &composed);
                    } else {
                        let mut composed_1c = [0u8; 256];
                        for i in 0..256 {
                            composed_1c[i] = lut_1c[auto_lut[i] as usize];
                        }
                        crate::transforms::runtime::lut::LutExecutor::apply(image, &composed_1c);
                    }
                } else {
                    let mut composed = [0u8; 256];
                    for i in 0..256 {
                        composed[i] = lut_1c[auto_lut[i] as usize];
                    }
                    crate::transforms::runtime::lut::LutExecutor::apply(image, &composed);
                }
                None
            }
            KernelKind::FusedMatrix(matrix) => {
                if image.channels == 3 {
                    crate::transforms::runtime::matrix::MatrixExecutor::apply(image, &matrix.matrix);
                }
                None
            }
            KernelKind::Single(op) | KernelKind::Barrier(op) => {
                crate::core::Executable::execute(op, image)
            }
        }
    }

    /// Construct a KernelKind from a slice of fused ops
    pub fn from_fused_ops(ops: &[SampledImageOp]) -> Self {
        if ops.is_empty() {
            return KernelKind::Single(SampledImageOp::Invert);
        }
        if ops.len() == 1 {
            return KernelKind::Single(ops[0].clone());
        }
        if ops.iter().all(|t| crate::exec_ir::optimizer::is_geometric_transform_sampled(t)) {
            let mut orientation = crate::transforms::Orientation::Identity;
            for op in ops {
                match op {
                    SampledImageOp::HorizontalFlip => orientation = orientation.compose(crate::transforms::Orientation::FlipH),
                    SampledImageOp::VerticalFlip => orientation = orientation.compose(crate::transforms::Orientation::FlipV),
                    SampledImageOp::Transpose => orientation = orientation.compose(crate::transforms::Orientation::Transpose),
                    SampledImageOp::Rotate { angle } => {
                        let a = match angle {
                            crate::sampled_ir::ops::RotateAngle::Rotate90 => crate::transforms::Orientation::Rot90,
                            crate::sampled_ir::ops::RotateAngle::Rotate180 => crate::transforms::Orientation::Rot180,
                            crate::sampled_ir::ops::RotateAngle::Rotate270 => crate::transforms::Orientation::Rot270,
                        };
                        orientation = orientation.compose(a);
                    }
                    _ => {}
                }
            }
            return KernelKind::Geometric(orientation);
        }
        if ops.iter().all(|t| t.is_lut_op()) {
            let fused_lut = crate::transforms::FusedLut::from_sampled_ops(ops);
            return KernelKind::FusedLut {
                luts_3c: fused_lut.luts_3c.map(Box::new),
                lut_1c: fused_lut.lut,
            };
        }
        KernelKind::Single(ops[0].clone())
    }
}

/// Execution IR node kind
///
/// The internal enum representing what kind of node this is.
/// NO RTTI - uses SampledImageOp enum directly.
#[derive(Clone)]
pub enum ExecNodeKind {
    /// A fused block of ops that execute as a single pass
    Fused(Vec<SampledImageOp>),

    /// A barrier that breaks fusion
    Barrier(SampledImageOp),
}

impl fmt::Debug for ExecNodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecNodeKind::Fused(ops) => f.debug_tuple("Fused").field(&ops.len()).finish(),
            ExecNodeKind::Barrier(_) => f.debug_tuple("Barrier").field(&"<op>").finish(),
        }
    }
}

/// Execution IR node
///
/// Represents either a fused block of ops or a barrier.
/// Dispatches via fast enum KernelKind without dynamic trait objects.
pub struct ExecNode {
    /// The kind of execution node (fused or barrier)
    pub kind: ExecNodeKind,

    /// Legacy fast-path kernel (when set, takes precedence)
    pub kernel: Option<FastKernel>,

    /// Zero-cost concrete kernel kind
    pub kernel_kind: Option<KernelKind>,
}

impl ExecNode {
    /// Create a new Fused node
    pub fn fused(ops: Vec<SampledImageOp>) -> Self {
        let kernel_kind = KernelKind::from_fused_ops(&ops);
        Self {
            kind: ExecNodeKind::Fused(ops),
            kernel: None,
            kernel_kind: Some(kernel_kind),
        }
    }

    /// Create a new Barrier node
    pub fn barrier(op: SampledImageOp) -> Self {
        let kernel_kind = KernelKind::Barrier(op.clone());
        Self {
            kind: ExecNodeKind::Barrier(op),
            kernel: None,
            kernel_kind: Some(kernel_kind),
        }
    }

    /// Create a node with a pre-bound legacy fast-path kernel
    pub fn with_kernel(kind: ExecNodeKind, kernel: FastKernel) -> Self {
        Self {
            kind,
            kernel: Some(kernel),
            kernel_kind: None,
        }
    }

    /// Create a node with a concrete zero-allocation kernel
    pub fn with_kernel_kind(kind: ExecNodeKind, kernel_kind: KernelKind) -> Self {
        Self {
            kind,
            kernel: None,
            kernel_kind: Some(kernel_kind),
        }
    }

    /// Execute this node on an image
    #[inline]
    pub fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        if let Some(ref k) = self.kernel_kind {
            return k.execute(image);
        }
        if let Some(ref k) = self.kernel {
            return k(image);
        }
        match &self.kind {
            ExecNodeKind::Fused(transforms) if transforms.len() == 1 => {
                crate::core::Executable::execute(&transforms[0], image)
            }
            ExecNodeKind::Barrier(op) => {
                crate::core::Executable::execute(op, image)
            }
            _ => None,
        }
    }

    /// Get the number of transforms in this node
    pub fn len(&self) -> usize {
        match &self.kind {
            ExecNodeKind::Fused(transforms) => transforms.len(),
            ExecNodeKind::Barrier(_) => 1,
        }
    }

    /// Is this node empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Is this node a fused block?
    pub fn is_fused(&self) -> bool {
        matches!(&self.kind, ExecNodeKind::Fused(_))
    }

    /// Is this node a barrier?
    pub fn is_barrier(&self) -> bool {
        matches!(&self.kind, ExecNodeKind::Barrier(_))
    }

    /// Get the name of this node type for debugging
    pub fn type_name(&self) -> &'static str {
        match &self.kind {
            ExecNodeKind::Fused(_) => "Fused",
            ExecNodeKind::Barrier(_) => "Barrier",
        }
    }

    /// Get a description of the transforms in this node
    pub fn describe(&self) -> String {
        match &self.kind {
            ExecNodeKind::Fused(transforms) => {
                format!("Fused({} transforms)", transforms.len())
            }
            ExecNodeKind::Barrier(_) => {
                format!("Barrier")
            }
        }
    }
}

impl fmt::Debug for ExecNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ExecNodeKind::Fused(transforms) => {
                f.debug_tuple("Fused").field(&transforms.len()).finish()
            }
            ExecNodeKind::Barrier(_) => f.debug_tuple("Barrier").field(&1).finish(),
        }
    }
}

/// Fusion statistics for an ExecPlan
#[derive(Debug, Clone)]
pub struct FusionStats {
    /// Statistics for each fusion block
    pub blocks: Vec<BlockStats>,
    /// Total number of input transforms
    pub total_input: usize,
    /// Total number of output execution nodes
    pub total_output: usize,
    /// Fusion ratio (input / output)
    pub fusion_ratio: f64,
}

impl FusionStats {
    /// Create new fusion statistics
    pub fn new(blocks: Vec<BlockStats>) -> Self {
        let total_input: usize = blocks.iter().map(|b| b.input_count).sum();
        let total_output: usize = blocks.iter().map(|b| b.output_count).sum();
        let fusion_ratio = if total_output > 0 {
            total_input as f64 / total_output as f64
        } else {
            1.0
        };

        Self {
            blocks,
            total_input,
            total_output,
            fusion_ratio,
        }
    }

    /// Print a formatted summary of the fusion statistics
    pub fn print(&self) {
        println!("\n=== Fusion Statistics ===");
        println!("Total input transforms: {}", self.total_input);
        println!("Total output exec nodes: {}", self.total_output);
        println!("Fusion ratio: {:.2}x", self.fusion_ratio);
        println!();

        // Count by strategy
        let mut strategy_counts = std::collections::HashMap::new();
        for stat in &self.blocks {
            *strategy_counts.entry(stat.strategy).or_insert(0) += 1;
        }

        if !strategy_counts.is_empty() {
            println!("Fusion strategies used:");
            for (strategy, count) in strategy_counts.iter() {
                println!("  {:?}: {} blocks", strategy, count);
            }
            println!();
        }

        // Detail each block
        if !self.blocks.is_empty() {
            println!("Block-by-block breakdown:");
            for (i, stat) in self.blocks.iter().enumerate() {
                println!(
                    "  Block {}: {} -> {} transforms ({:?})",
                    i, stat.input_count, stat.output_count, stat.strategy
                );
            }
            println!();
        }
    }

    /// Get a one-line summary
    pub fn summary(&self) -> String {
        format!(
            "{} -> {} transforms ({:.2}x fusion)",
            self.total_input, self.total_output, self.fusion_ratio
        )
    }
}

/// Optimized execution plan
///
/// This is the output of the optimizer. It represents the
/// most efficient way to execute the transforms, with fusion
/// applied where possible.
#[derive(Debug)]
pub struct ExecPlan {
    /// Optimized execution nodes
    pub nodes: Vec<ExecNode>,
    /// Fusion statistics (optional)
    stats: Option<FusionStats>,
}

impl ExecPlan {
    /// Create a new empty ExecPlan
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            stats: None,
        }
    }

    /// Create from a vector of exec nodes
    pub fn from_nodes(nodes: Vec<ExecNode>) -> Self {
        Self { nodes, stats: None }
    }

    /// Create from nodes with fusion statistics
    pub fn from_nodes_with_stats(nodes: Vec<ExecNode>, block_stats: Vec<BlockStats>) -> Self {
        let stats = FusionStats::new(block_stats);
        Self {
            nodes,
            stats: Some(stats),
        }
    }

    /// Number of execution nodes
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Is the plan empty?
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get an iterator over the nodes
    pub fn iter(&self) -> impl Iterator<Item = &ExecNode> {
        self.nodes.iter()
    }

    /// Count how many transforms are in fused blocks
    pub fn fused_transform_count(&self) -> usize {
        self.nodes
            .iter()
            .filter_map(|n| match &n.kind {
                ExecNodeKind::Fused(t) => Some(t.len()),
                _ => None,
            })
            .sum()
    }

    /// Count how many barriers exist
    pub fn barrier_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_barrier()).count()
    }

    /// Calculate the theoretical speedup from fusion
    ///
    /// Returns the ratio of original transforms to execution nodes.
    /// Higher values mean more fusion occurred.
    ///
    /// # Example
    /// If you have 10 transforms fused into 2 execution nodes,
    /// the fusion ratio is 5.0x.
    pub fn fusion_ratio(&self, original_transform_count: usize) -> f64 {
        if self.nodes.is_empty() {
            return 1.0;
        }
        original_transform_count as f64 / self.nodes.len() as f64
    }

    /// Get the fusion statistics (if available)
    pub fn stats(&self) -> Option<&FusionStats> {
        self.stats.as_ref()
    }

    /// Does this execution plan mutate its initial input buffer?
    ///
    /// Returns true if the FIRST execution node modifies the input buffer in-place.
    /// Returns false if the first node produces a new BarrierImage (e.g. Resize, Pad, Crop, Affine),
    /// which means subsequent transforms only mutate the new BarrierImage, leaving the caller's
    /// input buffer 100% untouched.
    pub fn mutates_input(&self) -> bool {
        if let Some(first_node) = self.nodes.first() {
            match &first_node.kind {
                ExecNodeKind::Barrier(_) => false,
                ExecNodeKind::Fused(ops) => {
                    if let Some(first_op) = ops.first() {
                        if matches!(first_op.access_pattern(), crate::core::AccessPattern::OutOfPlace) {
                            return false;
                        }
                    }
                    if let Some(last_op) = ops.last() {
                        if matches!(last_op, SampledImageOp::ToGray) {
                            return false;
                        }
                    }
                    !ops.is_empty()
                }
            }
        } else {
            false
        }
    }


    /// Print the fusion statistics (if available)
    pub fn print_stats(&self) {
        if let Some(stats) = &self.stats {
            stats.print();
        } else {
            println!("No fusion statistics available");
        }
    }

    /// Print a visual representation of the execution plan
    ///
    /// # Example Output
    /// ```text
    /// ExecPlan (3 nodes):
    ///   [0] Fused(3 transforms)
    ///   [1] Barrier(Resize)
    ///   [2] Fused(2 transforms)
    /// ```
    pub fn visualize(&self) {
        println!("\n=== ExecPlan Visualization ===");
        println!("Total nodes: {}", self.nodes.len());

        for (i, node) in self.nodes.iter().enumerate() {
            println!("  [{}] {} ({} transforms)", i, node.type_name(), node.len());

            // Print detailed transform names if verbose
            if node.len() <= 5 {
                let desc = node.describe();
                for line in desc.lines() {
                    println!("      {}", line);
                }
            }
        }

        if let Some(stats) = &self.stats {
            println!("\n  {}", stats.summary());
        }
        println!();
    }

    /// Get a detailed string representation of the plan
    pub fn detailed_description(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("ExecPlan ({} nodes):", self.nodes.len()));

        for (i, node) in self.nodes.iter().enumerate() {
            lines.push(format!(
                "  [{}] {} ({} transforms)",
                i,
                node.type_name(),
                node.len()
            ));

            let desc = node.describe();
            for line in desc.lines() {
                lines.push(format!("      {}", line));
            }
        }

        if let Some(stats) = &self.stats {
            lines.push(format!("\n  {}", stats.summary()));
        }

        lines.join("\n")
    }
}

impl fmt::Display for ExecPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExecPlan({} nodes)", self.nodes.len())
    }
}

impl Default for ExecPlan {
    fn default() -> Self {
        Self::new()
    }
}

// Execute methods are in execution.rs
impl ExecPlan {
    /// Execute the optimized plan on an image
    ///
    /// This executes the plan by:
    /// 1. For Fused nodes: executing all transforms in sequence via Executable
    /// 2. For Barrier nodes: executing the single transform via Executable
    ///
    /// When a transform returns Some(BarrierImage), the barrier image is used for
    /// subsequent operations. This handles Resize and other buffer-allocating ops.
    ///
    /// Returns Some(BarrierImage) if a transform allocated a new buffer, None otherwise.
    pub fn execute(
        &self,
        initial_image: &mut crate::core::FusableImage,
    ) -> Option<crate::core::BarrierImage> {
        super::execution::execute_plan(self, initial_image)
    }

    /// Execute the optimized plan on a batch of images efficiently
    ///
    /// This applies the same optimized single-image pipeline to each image in the batch.
    /// This is the first stage of the two-stage pipeline (single-image transforms → batch transforms).
    ///
    /// # Arguments
    /// - `images`: Mutable slice of BarrierImages to transform
    ///
    /// # Performance
    /// - Each image is processed independently with the same optimized plan
    /// - Fusion benefits (LUT, matrix, general) apply to each image
    /// - Use this before batch transforms (MixUp, CutMix, Mosaic)
    ///
    /// # Example
    /// ```ignore
    /// // Stage 1: Apply single-image transforms to batch
    /// exec_plan.execute_batch(&mut images);
    ///
    /// // Stage 2: Apply batch transforms
    /// let mut batch = Batch::new(images, labels);
    /// mixup.apply(&mut batch, &mut rng);
    /// ```
    pub fn execute_batch(&self, images: &mut [crate::core::BarrierImage]) {
        for img in images {
            let mut fusable = img.as_fusable();
            if let Some(new_barrier) = self.execute(&mut fusable) {
                // A barrier transform allocated a new buffer - update the image
                *img = new_barrier;
            }
            // If None, the transform was in-place and img is already modified
        }
    }
}
