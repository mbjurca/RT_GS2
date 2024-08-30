# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/1361/bin/cmake

# The command to remove a file.
RM = /snap/cmake/1361/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers

# Utility rule file for sibr_view_shaders.

# Include any custom commands dependencies for this target.
include src/core/view/CMakeFiles/sibr_view_shaders.dir/compiler_depend.make

# Include the progress variables for this target.
include src/core/view/CMakeFiles/sibr_view_shaders.dir/progress.make

sibr_view_shaders: src/core/view/CMakeFiles/sibr_view_shaders.dir/build.make
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/alpha_colored_mesh.frag /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/alpha_colored_mesh.frag
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/alpha_colored_mesh.vert /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/alpha_colored_mesh.vert
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/alpha_colored_per_triangle_normals.geom /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/alpha_colored_per_triangle_normals.geom
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/alpha_colored_per_triangle_normals.vert /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/alpha_colored_per_triangle_normals.vert
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/alpha_colored_per_vertex_normals.geom /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/alpha_colored_per_vertex_normals.geom
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/alpha_colored_per_vertex_normals.vert /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/alpha_colored_per_vertex_normals.vert
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/alpha_points.frag /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/alpha_points.frag
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/alpha_points.vert /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/alpha_points.vert
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/alpha_uv_tex.frag /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/alpha_uv_tex.frag
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/alpha_uv_tex_array.frag /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/alpha_uv_tex_array.frag
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/alphaimgview.fp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/alphaimgview.fp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/alphaimgview.vp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/alphaimgview.vp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/anaglyph.fp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/anaglyph.fp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/anaglyph.vp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/anaglyph.vp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/axisgizmo.fp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/axisgizmo.fp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/axisgizmo.vp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/axisgizmo.vp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/camstub.fp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/camstub.fp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/camstub.vp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/camstub.vp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/depth.fp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/depth.fp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/depth.vp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/depth.vp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/depthonly.fp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/depthonly.fp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/depthonly.vp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/depthonly.vp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/image_viewer.frag /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/image_viewer.frag
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/image_viewer.vert /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/image_viewer.vert
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/mesh_color.fp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/mesh_color.fp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/mesh_color.vp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/mesh_color.vp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/mesh_debugview.fp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/mesh_debugview.fp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/mesh_debugview.vp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/mesh_debugview.vp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/mesh_normal.fp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/mesh_normal.fp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/mesh_normal.vp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/mesh_normal.vp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/number.fp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/number.fp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/number.vp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/number.vp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/skybox.fp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/skybox.fp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/skybox.vp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/skybox.vp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/text-imgui.fp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/text-imgui.fp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/text-imgui.vp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/text-imgui.vp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/texture.fp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/texture.fp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/texture.vp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/texture.vp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/topview.fp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/topview.fp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/topview.vp /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/topview.vp
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && /snap/cmake/1361/bin/cmake -E copy_if_different /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/shaders/uv_mesh.vert /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/install/shaders/core/uv_mesh.vert
.PHONY : sibr_view_shaders

# Rule to build all files generated by this target.
src/core/view/CMakeFiles/sibr_view_shaders.dir/build: sibr_view_shaders
.PHONY : src/core/view/CMakeFiles/sibr_view_shaders.dir/build

src/core/view/CMakeFiles/sibr_view_shaders.dir/clean:
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view && $(CMAKE_COMMAND) -P CMakeFiles/sibr_view_shaders.dir/cmake_clean.cmake
.PHONY : src/core/view/CMakeFiles/sibr_view_shaders.dir/clean

src/core/view/CMakeFiles/sibr_view_shaders.dir/depend:
	cd /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view /home/mihnea/mihnea/gaussian-splatting/SIBR_viewers/src/core/view/CMakeFiles/sibr_view_shaders.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/core/view/CMakeFiles/sibr_view_shaders.dir/depend

