{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python312;

  # Path to local py-salt dependency (as string to avoid nix store copy)
  pySaltPath = "/home/arriburi/projects/py-salt-repo/py-salt";

in pkgs.mkShell {
  buildInputs = [
    python
    pkgs.python312Packages.pip
    pkgs.python312Packages.virtualenv

    # Development tools
    pkgs.ruff      # Python linter/formatter

    # System dependencies
    pkgs.ffmpeg
    pkgs.graphviz
    pkgs.portaudio
    pkgs.zlib

    # For building Python packages with C extensions
    pkgs.gcc
    pkgs.stdenv.cc.cc.lib
  ];

  shellHook = ''
    # Create virtual environment if it doesn't exist
    if [ ! -d .venv ]; then
      echo "Creating Python virtual environment..."
      ${python}/bin/python -m venv .venv
    fi

    # Activate virtual environment
    source .venv/bin/activate

    # Install dependencies from pyproject.toml
    echo "Installing dependencies..."
    pip install --upgrade pip

    # Install the project in editable mode (reads pyproject.toml)
    pip install -e .

    # Install development tools
    pip install visidata

    # Install local py-salt dependency
    if [ -d "${pySaltPath}" ]; then
      pip install -e ${pySaltPath}
    else
      echo "Warning: py-salt not found at ${pySaltPath}"
    fi

    echo ""
    echo "✓ Development environment ready!"
    echo "  Python version: $(python --version)"
    echo "  Project: urban-noise-classification"
    echo ""
    echo "Run 'python <script>.py' to execute your scripts"
    echo "Run 'exit' to leave this environment"
    echo ""
  '';

  # Set environment variables for libraries
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.portaudio}/lib:${pkgs.zlib}/lib";
}
