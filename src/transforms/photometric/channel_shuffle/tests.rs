// Tests for ChannelShuffle transform

use super::*;

#[test]
fn test_channel_shuffle_new() {
    let cs = ChannelShuffle::new(ChannelOrder::BGR);
    assert_eq!(cs.order, ChannelOrder::BGR);
}

#[test]
fn test_channel_shuffle_default() {
    let cs = ChannelShuffle::default();
    assert_eq!(cs.order, ChannelOrder::RGB);
}

#[test]
fn test_channel_shuffle_bgr() {
    let cs = ChannelShuffle::bgr();
    assert_eq!(cs.order, ChannelOrder::BGR);
}

#[test]
fn test_channel_shuffle_matrix_rgb() {
    let cs = ChannelShuffle::new(ChannelOrder::RGB);
    let matrix = cs.get_matrix();

    // Identity matrix
    assert_eq!(matrix[0][0], 1.0);
    assert_eq!(matrix[1][1], 1.0);
    assert_eq!(matrix[2][2], 1.0);
    assert!(matrix[0][1] == 0.0 && matrix[0][2] == 0.0);
}

#[test]
fn test_channel_shuffle_matrix_bgr() {
    let cs = ChannelShuffle::new(ChannelOrder::BGR);
    let matrix = cs.get_matrix();

    // R' = B, G' = G, B' = R
    assert_eq!(matrix[0][2], 1.0); // R' = B
    assert_eq!(matrix[1][1], 1.0); // G' = G
    assert_eq!(matrix[2][0], 1.0); // B' = R
}

#[test]
fn test_channel_shuffle_execute_rgb() {
    // RGB order should leave image unchanged
    let mut data = vec![100u8, 150, 200];
    let mut img = FusableImage::new(&mut data, 1, 1, 3);

    ChannelShuffle::new(ChannelOrder::RGB).execute(&mut img);

    assert_eq!(img.data[0], 100);
    assert_eq!(img.data[1], 150);
    assert_eq!(img.data[2], 200);
}

#[test]
fn test_channel_shuffle_execute_bgr() {
    // BGR should swap R and B
    let mut data = vec![100u8, 150, 200];
    let mut img = FusableImage::new(&mut data, 1, 1, 3);

    ChannelShuffle::new(ChannelOrder::BGR).execute(&mut img);

    assert_eq!(img.data[0], 200); // R' = original B
    assert_eq!(img.data[1], 150); // G' = original G
    assert_eq!(img.data[2], 100); // B' = original R
}

#[test]
fn test_channel_shuffle_execute_grb() {
    // GRB should swap R and G
    let mut data = vec![100u8, 150, 200];
    let mut img = FusableImage::new(&mut data, 1, 1, 3);

    ChannelShuffle::new(ChannelOrder::GRB).execute(&mut img);

    assert_eq!(img.data[0], 150); // R' = original G
    assert_eq!(img.data[1], 100); // G' = original R
    assert_eq!(img.data[2], 200); // B' = original B
}

#[test]
fn test_channel_shuffle_execute_gbr() {
    let mut data = vec![100u8, 150, 200];
    let mut img = FusableImage::new(&mut data, 1, 1, 3);

    ChannelShuffle::new(ChannelOrder::GBR).execute(&mut img);

    assert_eq!(img.data[0], 150); // R' = original G
    assert_eq!(img.data[1], 200); // G' = original B
    assert_eq!(img.data[2], 100); // B' = original R
}

#[test]
fn test_channel_shuffle_execute_rbg() {
    let mut data = vec![100u8, 150, 200];
    let mut img = FusableImage::new(&mut data, 1, 1, 3);

    ChannelShuffle::new(ChannelOrder::RBG).execute(&mut img);

    assert_eq!(img.data[0], 100); // R' = original R
    assert_eq!(img.data[1], 200); // G' = original B
    assert_eq!(img.data[2], 150); // B' = original G
}

#[test]
fn test_channel_shuffle_execute_brg() {
    let mut data = vec![100u8, 150, 200];
    let mut img = FusableImage::new(&mut data, 1, 1, 3);

    ChannelShuffle::new(ChannelOrder::BRG).execute(&mut img);

    assert_eq!(img.data[0], 200); // R' = original B
    assert_eq!(img.data[1], 100); // G' = original R
    assert_eq!(img.data[2], 150); // B' = original G
}

#[test]
fn test_channel_shuffle_grayscale_passthrough() {
    let mut data = vec![128u8];
    let mut img = FusableImage::new(&mut data, 1, 1, 1);

    ChannelShuffle::bgr().execute(&mut img);

    assert_eq!(img.data[0], 128);
}

#[test]
fn test_channel_shuffle_access_pattern() {
    let _cs = ChannelShuffle::bgr();
    assert_eq!(_cs.access(), AccessPattern::InPlace);
    assert_eq!(_cs.shape_effect(), ShapeEffect::Preserve);
}

#[test]
fn test_channel_order_all() {
    let all = ChannelOrder::all();
    assert_eq!(all.len(), 6);
    assert!(all.contains(&ChannelOrder::RGB));
    assert!(all.contains(&ChannelOrder::BGR));
    assert!(all.contains(&ChannelOrder::GRB));
    assert!(all.contains(&ChannelOrder::GBR));
    assert!(all.contains(&ChannelOrder::RBG));
    assert!(all.contains(&ChannelOrder::BRG));
}

#[test]
fn test_channel_shuffle_all_orders_unique() {
    use std::collections::HashSet;
    let all = ChannelOrder::all();
    let set: HashSet<_> = all.iter().collect();
    assert_eq!(set.len(), 6); // All unique
}
