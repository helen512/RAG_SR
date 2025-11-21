#!/bin/bash
# Script to run Reacher visualization with proper OpenGL setup for WSL2
# This bypasses conda library conflicts by using system libraries for OpenGL

echo "Setting up OpenGL environment for WSL2..."

# Set up environment to use system OpenGL libraries instead of conda's
export MUJOCO_GL=glfw
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3

# Use system's libstdc++ instead of conda's (this is the key fix)
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

echo "Environment configured:"
echo "  MUJOCO_GL: $MUJOCO_GL"
echo "  LIBGL_ALWAYS_SOFTWARE: $LIBGL_ALWAYS_SOFTWARE"
echo "  LIBGL_DRIVERS_PATH: $LIBGL_DRIVERS_PATH"
echo "  Using system libstdc++ to fix GLIBCXX conflicts"

echo -e "\nRunning Reacher visualization..."
conda run -n safe python "$@"
