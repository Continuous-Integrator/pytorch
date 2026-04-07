# Post-build steps previously handled by setup.py's build_ext.run().
# These run as CMake install(SCRIPT) or install(CODE) commands.

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()
if(NOT TORCH_INSTALL_INCLUDE_DIR)
  set(TORCH_INSTALL_INCLUDE_DIR include)
endif()

# --- Header wrapping with TORCH_STABLE_ONLY guards ---
# Wrap installed headers so they error when included with TORCH_STABLE_ONLY
# or TORCH_TARGET_VERSION defined. This is done at install time via a script.
install(CODE "
  set(_include_dir \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_INCLUDE_DIR}\")
  if(EXISTS \"\${_include_dir}\")
    message(STATUS \"Wrapping headers with TORCH_STABLE_ONLY guards...\")
    set(_header_extensions h hpp cuh)
    set(_exclude_patterns
      \"torch/headeronly/\"
      \"torch/csrc/stable/\"
      \"torch/csrc/inductor/aoti_torch/c/\"
      \"torch/csrc/inductor/aoti_torch/generated/\"
    )
    set(_wrap_marker \"#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)\")

    foreach(_ext IN ITEMS h hpp cuh)
      file(GLOB_RECURSE _headers \"\${_include_dir}/*.\${_ext}\")
      foreach(_header IN LISTS _headers)
        file(RELATIVE_PATH _rel \"\${_include_dir}\" \"\${_header}\")

        # Check exclusion patterns
        set(_excluded FALSE)
        foreach(_pat IN LISTS _exclude_patterns)
          string(FIND \"\${_rel}\" \"\${_pat}\" _pos)
          if(NOT _pos EQUAL -1)
            set(_excluded TRUE)
            break()
          endif()
        endforeach()
        if(_excluded)
          continue()
        endif()

        file(READ \"\${_header}\" _content)
        string(FIND \"\${_content}\" \"\${_wrap_marker}\" _already_wrapped)
        if(_already_wrapped EQUAL 0)
          continue()
        endif()

        set(_wrapped \"\${_wrap_marker}\\n\${_content}\\n#else\\n\")
        string(APPEND _wrapped
          \"#error \\\"This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined.\\\"\\n\")
        string(APPEND _wrapped
          \"#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)\\n\")
        file(WRITE \"\${_header}\" \"\${_wrapped}\")
      endforeach()
    endforeach()
  endif()
")

# --- Compile commands merging ---
# Merge compile_commands.json from build subdirectories.
# Write the script to a file to avoid CMake stripping newlines from multiline
# command arguments when passed through Ninja.
file(WRITE "${CMAKE_BINARY_DIR}/merge_compile_commands.py"
"import json, pathlib, itertools\n\
build = pathlib.Path('${CMAKE_BINARY_DIR}')\n\
ninja = list(build.glob('*compile_commands.json'))\n\
cmake_sub = list((build / 'torch' / 'lib' / 'build').glob('*/compile_commands.json')) if (build / 'torch' / 'lib' / 'build').exists() else []\n\
cmds = [e for f in itertools.chain(ninja, cmake_sub) for e in json.loads(f.read_text())]\n\
for c in cmds:\n\
    if c.get('command', '').startswith('gcc '):\n\
        c['command'] = 'g++ ' + c['command'][4:]\n\
out = pathlib.Path('${PROJECT_SOURCE_DIR}/compile_commands.json')\n\
new = json.dumps(cmds, indent=2)\n\
if not out.exists() or out.read_text() != new:\n\
    out.write_text(new)\n\
")
add_custom_target(merge_compile_commands ALL
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_BINARY_DIR}/merge_compile_commands.py"
  COMMENT "Merging compile_commands.json..."
  VERBATIM
)
