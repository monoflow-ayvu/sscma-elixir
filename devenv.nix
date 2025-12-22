{ pkgs, config, ... }:

{
  # Basic packages needed for the development environment
  packages = with pkgs; [
    # Build essentials
    cmake
    gnumake
    gcc
    pkg-config
    ninja

    # Additional tools
    git
    wget
    curl
    unzip

    # Python and requirements
    python3
    python3Packages.pip

    # Tools for SDK extraction
    gnutar
    gzip
    findutils

    # Basic system tools
    coreutils
    bashInteractive
    which

    # Libraries needed by cross-compiler
    zlib
    ncurses
    glibc
  ];

  # Environment variables with absolute paths
  env = {
    SG200X_FHS = "1";
    CMAKE_TOOLCHAIN_FILE = "${config.env.DEVENV_ROOT}/cmake/toolchain-riscv64-linux-musl-x86_64.cmake";
    SG200X_SDK_PATH = "${config.env.DEVENV_ROOT}/.devenv/state/sg200x-sdk/sg2002_recamera_emmc";
  };

  # Task to setup SDK before entering shell
  tasks."sg200x:setup-sdk" = {
    exec = ''
      # Setup SDK and host-tools if not present
      RECAMERA_ROOT="${config.env.DEVENV_ROOT}/.devenv/state/sg200x-sdk"

      if [ ! -d "$RECAMERA_ROOT/sg2002_recamera_emmc" ]; then
        echo "📦 Setting up SDK and host-tools..."
        mkdir -p "$RECAMERA_ROOT"

        # Download and extract SDK
        echo "📁 Downloading and extracting SDK..."
        if [ ! -f "$RECAMERA_ROOT/reCameraOS_sdk_v0.2.0.tar.gz" ]; then
          wget -O "$RECAMERA_ROOT/reCameraOS_sdk_v0.2.0.tar.gz" \
            "https://github.com/Seeed-Studio/reCamera-OS/releases/download/0.2.0/reCameraOS_sdk_v0.2.0.tar.gz"
        fi

        tar -xzf "$RECAMERA_ROOT/reCameraOS_sdk_v0.2.0.tar.gz" -C "$RECAMERA_ROOT" --strip-components=0

        # Remove broken symlinks
        echo "Cleaning up broken symlinks..."
        find "$RECAMERA_ROOT" -type l -exec test ! -e {} \; -print | while read link; do
          echo "Removing broken symlink: $link"
          rm -f "$link"
        done

        # Clone host-tools
        echo "🔧 Cloning host-tools..."
        git clone --verbose --progress https://github.com/sophgo/host-tools.git "$RECAMERA_ROOT/host-tools"

        echo "✅ SDK setup complete!"
      else
        echo "✅ SDK already set up in $RECAMERA_ROOT"
      fi
    '';
    before = [ "devenv:enterShell" ];
  };

  # Shell initialization script (runs after SDK setup)
  enterShell = ''
    # Set up the environment variables with absolute paths
    RECAMERA_ROOT="${config.env.DEVENV_ROOT}/.devenv/state/sg200x-sdk"
    export SG200X_SDK_PATH="$RECAMERA_ROOT/sg2002_recamera_emmc"

    # Add cross-compiler to PATH with absolute path
    COMPILER_DIR="$RECAMERA_ROOT/host-tools/gcc/riscv64-linux-musl-x86_64/bin"
    export PATH="$COMPILER_DIR:$PATH"

    # Create symlinks for the toolchain naming convention
    if [ -d "$COMPILER_DIR" ] && [ ! -f "$COMPILER_DIR/riscv64-unknown-linux-musl-gcc" ]; then
      echo "🔗 Creating compiler symlinks for toolchain compatibility..."
      for tool in gcc g++ objcopy objdump ar as ld nm ranlib strip; do
        if [ -f "$COMPILER_DIR/riscv64-linux-musl-$tool" ]; then
          ln -sf "riscv64-linux-musl-$tool" "$COMPILER_DIR/riscv64-unknown-linux-musl-$tool"
        fi
      done
    fi

    # Set explicit compiler environment variables with absolute paths
    export CC="$COMPILER_DIR/riscv64-unknown-linux-musl-gcc"
    export CXX="$COMPILER_DIR/riscv64-unknown-linux-musl-g++"
    export AR="$COMPILER_DIR/riscv64-unknown-linux-musl-ar"
    export STRIP="$COMPILER_DIR/riscv64-unknown-linux-musl-strip"

    echo "🚀 SG200X devenv Development Environment Ready!"
    echo "📁 SDK Path: $SG200X_SDK_PATH"
    echo "🔧 Toolchain File: ${config.env.CMAKE_TOOLCHAIN_FILE}"
    echo "🔧 Compiler Dir: $COMPILER_DIR"
    echo ""
    echo "💡 To configure the project, run:"
    echo "   cmake -B build -S ."
    echo "🛠️ To build the project, run:"
    echo "   cmake --build build --config Release"
    echo ""
    echo "📝 Or use the convenience scripts:"
    echo "   devenv run prepare"
    echo "   devenv run build"
  '';

  scripts.prepare.exec = ''
    # Prepare the project
    echo "🔧 Preparing the project..."
    cmake -B build -S .
  '';

  scripts.build.exec = ''
    # Build the project
    echo "🛠️ Building the project..."
    cmake --build build --config Release
  '';
}
