{ pkgs ? import <nixpkgs> {} }:

let
  # Download the reCamera SDK
  reCameraSDK = pkgs.fetchurl {
    url = "https://github.com/Seeed-Studio/reCamera-OS/releases/download/0.2.0/reCameraOS_sdk_v0.2.0.tar.gz";
    sha256 = "75f6aa5e6cf164bf309755b6870e5d35f83696a3cf274dc47e865e8450446f08";
  };

  # Create a derivation that sets up the SDK
  setupSDK = pkgs.stdenv.mkDerivation {
    name = "sg200x-sdk-setup";
    src = reCameraSDK;
    
    buildInputs = [ pkgs.gnutar pkgs.gzip pkgs.findutils ];
    
    # Allow broken symlinks since the SDK contains some that point to runtime paths
    dontFixup = true;
    
    installPhase = ''
      mkdir -p $out
      tar -xzf $src -C $out --strip-components=0
      
      # Remove obviously broken symlinks that cause issues
      echo "Cleaning up broken symlinks..."
      find $out -type l -exec test ! -e {} \; -print | while read link; do
        echo "Removing broken symlink: $link"
        rm -f "$link"
      done
    '';
  };

  # Create the FHS environment
  fhsEnv = pkgs.buildFHSEnv {
    name = "sg200x-dev";
    
    targetPkgs = pkgs: (with pkgs; [
      # Build essentials
      cmake
      gnumake
      gcc
      pkg-config
      
      # Additional tools that might be needed
      git
      wget
      curl
      unzip
      
      # Python and requirements (if needed by the project)
      python3
      python3Packages.pip
      
      # Tools for SDK extraction
      gnutar
      gzip
      findutils
      
      # Basic system tools
      coreutils
      bash
      which
      
      # Libraries needed by cross-compiler
      zlib
      ncurses
      glibc
    ]);

    multiPkgs = pkgs: with pkgs; [
      # Libraries needed by the cross-compiler toolchain
      zlib
      ncurses
      stdenv.cc.cc.lib
    ];

    extraBwrapArgs = [
      # No special mounts needed - we'll copy SDK during setup
    ];

    profile = ''
      # Setup SDK and host-tools if not present
      RECAMERA_ROOT="$HOME/.nix-fhs-recamera"
      if [ ! -d "$RECAMERA_ROOT/sg2002_recamera_emmc" ]; then
        echo "📦 Setting up SDK and host-tools..."
        mkdir -p "$RECAMERA_ROOT"
        
        # Copy SDK from nix store
        echo "📁 Copying SDK..."
        cp -r ${setupSDK}/* "$RECAMERA_ROOT/"
        chmod -R +w "$RECAMERA_ROOT"
        
        # Clone host-tools
        echo "🔧 Cloning host-tools..."
        git clone https://github.com/sophgo/host-tools.git "$RECAMERA_ROOT/host-tools"
      fi
      
      # Set up the environment variables
      export SG200X_SDK_PATH="$RECAMERA_ROOT/sg2002_recamera_emmc"
      export CMAKE_TOOLCHAIN_FILE="cmake/toolchain-riscv64-linux-musl-x86_64.cmake"
      
      # Add cross-compiler to PATH
      export PATH="$RECAMERA_ROOT/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH"
      
      # Create symlinks for the toolchain naming convention
      COMPILER_DIR="$RECAMERA_ROOT/host-tools/gcc/riscv64-linux-musl-x86_64/bin"
      if [ -d "$COMPILER_DIR" ] && [ ! -f "$COMPILER_DIR/riscv64-unknown-linux-musl-gcc" ]; then
        echo "🔗 Creating compiler symlinks for toolchain compatibility..."
        for tool in gcc g++ objcopy objdump ar as ld nm ranlib strip; do
          if [ -f "$COMPILER_DIR/riscv64-linux-musl-$tool" ]; then
            ln -sf "riscv64-linux-musl-$tool" "$COMPILER_DIR/riscv64-unknown-linux-musl-$tool"
          fi
        done
      fi
      
      # Set explicit compiler environment variables
      export CC="riscv64-unknown-linux-musl-gcc"
      export CXX="riscv64-unknown-linux-musl-g++"
      export AR="riscv64-unknown-linux-musl-ar"
      export STRIP="riscv64-unknown-linux-musl-strip"
      
      echo "🚀 SG200X FHS Development Environment Ready!"
      echo "📁 SDK Path: $SG200X_SDK_PATH"
      echo "🔧 Cross-compiler tools available in PATH"
      echo "🛠️  CMake Toolchain: $CMAKE_TOOLCHAIN_FILE"
      echo "📋 CC: $CC ($(which $CC 2>/dev/null || echo 'not found'))"
      echo "📋 CXX: $CXX ($(which $CXX 2>/dev/null || echo 'not found'))"
      echo ""
      echo "💡 To configure the project, run:"
      echo "   cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-linux-musl-x86_64.cmake"
      echo ""
      echo "🏠 Current directory: $(pwd)"
      echo "🌐 FHS Environment: /usr, /bin, /lib are available as expected"
      echo "📁 SDK available at: $SG200X_SDK_PATH"
    '';
  };

in pkgs.mkShell {
  buildInputs = [ ];
  
  shellHook = ''
    echo "🔄 Entering SG200X FHS Development Environment..."
    echo "🚀 Starting FHS environment with SDK and cross-compiler tools..."
    echo ""
    
    # Execute the FHS environment
    exec ${fhsEnv}/bin/sg200x-dev
  '';
}