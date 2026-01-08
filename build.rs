fn main() {
    #[cfg(feature = "opencv-static")]
    {
        // Static linking for minimal OpenCV (core + imgproc)
        // Build OpenCV using scripts/build_opencv_static.sh first

        // Get the install directory from environment or use default
        let static_dir = std::env::var("OPENCV_STATIC_DIR").unwrap_or_else(|_| {
            // Try default location relative to project root
            let default = format!(
                "{}/opencv-static/build",
                std::env::var("CARGO_MANIFEST_DIR").unwrap()
            );
            if std::path::Path::new(&default).exists() {
                default
            } else {
                // Fallback to /usr/local
                "/usr/local".to_string()
            }
        });

        let lib_path = format!("{}/lib", static_dir);
        let include_path = format!("{}/include/opencv4", static_dir);

        // Verify paths exist
        if !std::path::Path::new(&lib_path).exists() {
            panic!(
                "OpenCV static lib directory not found at: {}\n\
                 Please run: ./scripts/build_opencv_static.sh\n\
                 Or set OPENCV_STATIC_DIR to your OpenCV install path",
                lib_path
            );
        }

        // Check if environment is configured
        if std::env::var("OPENCV_LINK_STATIC").is_err() {
            println!("cargo:warning=⚠️  OPENCV_LINK_STATIC is not set!");
            println!("cargo:warning=The 'opencv' dependency will likely link DYNAMICALLY.");
            println!("cargo:warning=Please source scripts/setup_static_opencv.sh before building.");
        }

        // Explicit static linking directives for rustc (this crate)
        // Note: The 'opencv' crate must ALSO be configured via env vars to link statically
        println!("cargo:rustc-link-search=native={}", lib_path);
        println!("cargo:rustc-link-lib=static=opencv_imgproc");
        println!("cargo:rustc-link-lib=static=opencv_core");
        // Link core again due to circular dependencies/ordering in some static builds
        println!("cargo:rustc-link-lib=static=opencv_core");

        // Link Mac OS frameworks commonly required by OpenCV
        #[cfg(target_os = "macos")]
        {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }

        println!(
            "cargo:warning=OpenCV static linking requested. Libs at: {}",
            static_dir
        );
    }

    #[cfg(all(feature = "opencv", not(feature = "opencv-static")))]
    {
        // Dynamic linking for OpenCV (from conda or system)

        // If OPENCV_LINK_LIBS is set, the opencv crate will use it directly
        // Otherwise, we can set CONDA_PREFIX-based hints for the opencv crate
        if std::env::var("OPENCV_LINK_LIBS").is_err() {
            // Try to find conda environment
            if let Ok(conda_prefix) = std::env::var("CONDA_PREFIX") {
                // Check if this is a named environment (has /envs/ in path)
                let lib_path = if conda_prefix.contains("/envs/") {
                    // Already pointing to an environment, use directly
                    format!("{}/lib", conda_prefix)
                } else {
                    // Base conda, check if we're in a named environment
                    std::env::var("CONDA_DEFAULT_ENV")
                        .map(|env_name| format!("{}/envs/{}/lib", conda_prefix, env_name))
                        .unwrap_or_else(|_| format!("{}/lib", conda_prefix))
                };

                let include_path = lib_path.replace("/lib", "/include/opencv4");

                // Only set if paths exist
                if std::path::Path::new(&lib_path).exists() {
                    println!("cargo:rustc-env=OPENCV_LINK_PATHS={}", lib_path);
                    println!("cargo:rustc-env=OPENCV_INCLUDE_PATHS={}", include_path);
                }
            }
        }

        // Explicit linking - the opencv crate handles this, but we can
        // ensure the libraries are linked for the imgproc feature
        println!("cargo:rustc-link-lib=opencv_core");
        println!("cargo:rustc-link-lib=opencv_imgproc");
    }
}
