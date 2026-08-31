// Execution IR nodes
//
// Defines ExecNode and ExecPlan types for the optimized execution representation.

use crate::core::{BarrierImage, FusableImage};
use crate::exec_ir::optimizer::BlockStats;
use crate::sampled_ir::SampledImageOp;
use std::fmt;

/// Fast-path kernel function pointer
///
/// Pre-bound function that executes a transform without type checks.
/// This is the "zero-cost abstraction" path that avoids dynamic dispatch.
pub type FastKernel = Box<dyn Fn(&mut FusableImage) -> Option<BarrierImage> + Send + Sync>;

/// Execution IR node kind
///
/// The internal enum representing what kind of node this is.
/// NO RTTI - uses SampledImageOp enum directly.
pub enum ExecNodeKind {
    /// A fused block of ops that execute as a single pass
    ///
    /// All ops in this block are InPlace + Preserve,
    /// so they can be safely fused into one loop over pixels.
    Fused(Vec<SampledImageOp>),

    /// A barrier that breaks fusion
    ///
    /// Barriers are transforms that change shape (Resize, Crop, Pad)
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
/// This is the output of the optimizer.
///
/// NO RTTI - uses SampledImageOp enum with match dispatch.
pub struct ExecNode {
    /// The kind of execution node (fused or barrier)
    pub kind: ExecNodeKind,

    /// Pre-bound fast-path kernel (set during optimization)
    ///
    /// When Some: executes directly without type checks
    /// When None: uses match dispatch on SampledImageOp (still no RTTI!)
    pub kernel: Option<FastKernel>,
}

impl ExecNode {
    /// Create a new Fused node
    pub fn fused(ops: Vec<SampledImageOp>) -> Self {
        Self {
            kind: ExecNodeKind::Fused(ops),
            kernel: None,
        }
    }

    /// Create a new Barrier node
    pub fn barrier(op: SampledImageOp) -> Self {
        Self {
            kind: ExecNodeKind::Barrier(op),
            kernel: None,
        }
    }

    /// Create a node with a pre-bound fast-path kernel
    pub fn with_kernel(kind: ExecNodeKind, kernel: FastKernel) -> Self {
        Self {
            kind,
            kernel: Some(kernel),
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
                ExecNodeKind::Fused(ops) => !ops.is_empty(),
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
