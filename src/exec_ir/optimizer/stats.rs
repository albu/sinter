// Statistics and debug output for the optimizer

use std::collections::HashMap;

/// Which fusion strategy was applied
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionStrategy {
    /// No fusion (empty block or barrier)
    None,
    /// Geometric-only fusion (D4 group composition)
    Geometric,
    /// Structural fusion (photometric lifting into geometric kernels)
    Structural,
    /// Matrix fusion (3x3 RGB matrix composition)
    Matrix,
    /// Pure LUT fusion (composed LUT)
    Lut,
    /// General PixelOp fusion (fallback)
    General,
    /// Identity transforms (skipped)
    Identity,
}

/// Fusion statistics for a single block
#[derive(Debug, Clone)]
pub struct BlockStats {
    /// Number of input transforms
    pub input_count: usize,
    /// Number of output execution nodes
    pub output_count: usize,
    /// Which fusion strategy was used
    pub strategy: FusionStrategy,
}

impl Default for BlockStats {
    fn default() -> Self {
        Self {
            input_count: 0,
            output_count: 0,
            strategy: FusionStrategy::None,
        }
    }
}

/// Debug/logging configuration for the optimizer
#[derive(Debug, Clone, Copy)]
pub enum OptimizerDebug {
    /// No debug output
    None,
    /// Log fusion decisions
    Verbose,
}

impl Default for OptimizerDebug {
    fn default() -> Self {
        Self::None
    }
}

/// Print a summary of fusion statistics
pub fn print_stats(stats: &[BlockStats]) {
    if stats.is_empty() {
        println!("No fusion statistics available (run optimize() first)");
        return;
    }

    let total_input: usize = stats.iter().map(|s| s.input_count).sum();
    let total_output: usize = stats.iter().map(|s| s.output_count).sum();
    let fusion_ratio = total_input as f64 / total_output.max(1) as f64;

    println!("\n=== Fusion Statistics ===");
    println!("Total input transforms: {}", total_input);
    println!("Total output exec nodes: {}", total_output);
    println!("Fusion ratio: {:.2}x", fusion_ratio);
    println!();

    // Count by strategy
    let mut strategy_counts = HashMap::new();
    for stat in stats {
        *strategy_counts.entry(stat.strategy).or_insert(0) += 1;
    }

    println!("Fusion strategies used:");
    for (strategy, count) in strategy_counts.iter() {
        println!("  {:?}: {} blocks", strategy, count);
    }
    println!();

    // Detail each block
    println!("Block-by-block breakdown:");
    for (i, stat) in stats.iter().enumerate() {
        println!("  Block {}: {} -> {} transforms ({:?})",
            i, stat.input_count, stat.output_count, stat.strategy);
    }
    println!();
}
